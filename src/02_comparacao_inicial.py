r"""Modelo 03 do TCC: busca SARIMA ampliada e comparacao dos tres metodos.

Uso no PowerShell (arquivo da base na mesma pasta):
    python modelo_03_sarima_busca_ampliada.py

Ou informando o caminho da base:
    python modelo_03_sarima_busca_ampliada.py "C:\\caminho\\Base_Oficial_....xlsx"

O script usa somente o conjunto de treinamento para escolher o SARIMA. Primeiro
exige convergencia e residuos sem autocorrelacao significativa nos lags 12 e 24;
entre os candidatos aprovados, escolhe o menor AIC. Os 12 meses finais sao usados
uma unica vez para a avaliacao externa.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX


N_TESTE = 12
SAZONALIDADE = 12


def localizar_base(caminho_informado: str | None) -> Path:
    if caminho_informado:
        base = Path(caminho_informado).expanduser().resolve()
        if not base.exists():
            raise FileNotFoundError(f"Base nao encontrada: {base}")
        return base

    pasta = Path(__file__).resolve().parent
    candidatos = sorted(pasta.glob("Base_Oficial_Filtrada_Consumo_Residencial*.xlsx"))
    if len(candidatos) == 1:
        return candidatos[0]
    if not candidatos:
        raise FileNotFoundError(
            "Coloque a planilha Base_Oficial_Filtrada_Consumo_Residencial...xlsx "
            "na mesma pasta do script ou informe o caminho como argumento."
        )
    raise RuntimeError("Ha mais de uma base candidata na pasta; informe o caminho correto.")


def carregar_serie(base: Path) -> pd.Series:
    df = pd.read_excel(base, sheet_name="Residencial Brasil")
    obrigatorias = {"Data", "Consumo_GWh"}
    ausentes = obrigatorias.difference(df.columns)
    if ausentes:
        raise ValueError(f"Colunas ausentes na base: {sorted(ausentes)}")

    dados = df.loc[:, ["Data", "Consumo_GWh"]].copy()
    dados["Data"] = pd.to_datetime(dados["Data"], errors="raise")
    dados["Consumo_GWh"] = pd.to_numeric(dados["Consumo_GWh"], errors="raise")
    dados = dados.sort_values("Data")

    if dados["Data"].duplicated().any():
        raise ValueError("A serie possui datas duplicadas.")
    if dados["Consumo_GWh"].isna().any() or (dados["Consumo_GWh"] <= 0).any():
        raise ValueError("A serie possui consumo ausente, zero ou negativo.")

    indice_esperado = pd.date_range(dados["Data"].min(), dados["Data"].max(), freq="MS")
    if not dados["Data"].reset_index(drop=True).equals(pd.Series(indice_esperado)):
        raise ValueError("A serie nao e mensal completa; existem lacunas ou datas irregulares.")

    serie = pd.Series(
        dados["Consumo_GWh"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(dados["Data"], freq="MS"),
        name="Consumo_GWh",
    )
    if len(serie) != 270 or serie.index.min() != pd.Timestamp("2004-01-01") or serie.index.max() != pd.Timestamp("2026-06-01"):
        raise ValueError(
            "A base nao corresponde ao recorte aprovado (jan/2004 a jun/2026, 270 meses)."
        )
    return serie


def metricas(real: pd.Series, previsao: pd.Series) -> dict[str, float]:
    y = real.to_numpy(dtype=float)
    p = previsao.to_numpy(dtype=float)
    erro = y - p
    return {
        "MAE_GWh": float(np.mean(np.abs(erro))),
        "RMSE_GWh": float(np.sqrt(np.mean(erro**2))),
        "MAPE_percentual": float(np.mean(np.abs(erro / y)) * 100),
        "Bias_real_menos_previsto_GWh": float(np.mean(erro)),
    }


def ajustar_holt_winters(treino: pd.Series) -> tuple[object, pd.DataFrame]:
    candidatos = []
    ajustes = {}
    for sazonal in ("add", "mul"):
        nome = "Aditiva" if sazonal == "add" else "Multiplicativa"
        try:
            ajuste = ExponentialSmoothing(
                treino,
                trend="add",
                seasonal=sazonal,
                seasonal_periods=SAZONALIDADE,
                damped_trend=False,
                initialization_method="estimated",
            ).fit(optimized=True, remove_bias=False)
            ajustes[nome] = ajuste
            candidatos.append(
                {
                    "Modelo": nome,
                    "SSE_treinamento": float(ajuste.sse),
                    "AIC": float(ajuste.aic),
                    "Status": "OK",
                }
            )
        except Exception as exc:
            candidatos.append(
                {
                    "Modelo": nome,
                    "SSE_treinamento": np.nan,
                    "AIC": np.nan,
                    "Status": f"ERRO: {type(exc).__name__}: {exc}",
                }
            )
    tabela = pd.DataFrame(candidatos)
    validos = tabela[tabela["Status"] == "OK"]
    if validos.empty:
        raise RuntimeError("Nenhum Holt-Winters foi ajustado com sucesso.")
    escolhido = validos.sort_values("SSE_treinamento").iloc[0]["Modelo"]
    return ajustes[escolhido], tabela


def diagnostico_residuos(ajuste: object, ordem: tuple, ordem_sazonal: tuple) -> pd.DataFrame:
    """Ljung-Box sobre erros padronizados, excluindo o periodo de inicializacao."""
    erros = np.asarray(ajuste.filter_results.standardized_forecasts_error)[0]
    burn = int(
        getattr(
            ajuste,
            "loglikelihood_burn",
            getattr(ajuste.model, "loglikelihood_burn", 0),
        )
    )
    erros = erros[burn:]
    erros = erros[np.isfinite(erros)]
    graus_modelo = ordem[0] + ordem[2] + ordem_sazonal[0] + ordem_sazonal[2]
    return acorr_ljungbox(
        erros,
        lags=[12, 24],
        model_df=graus_modelo,
        return_df=True,
    ).reset_index(names="Lag")


def busca_sarima(treino: pd.Series) -> tuple[object, pd.DataFrame, dict]:
    # Grade ampliada: 81 modelos, todos com diferenca regular e sazonal.
    ordens = [(p, 1, q) for p in (0, 1, 2) for q in (0, 1, 2)]
    sazonais = [(P, 1, Q, SAZONALIDADE) for P in (0, 1, 2) for Q in (0, 1, 2)]
    resultados = []
    melhor_geral = None
    melhor_aic_geral = np.inf
    melhor_aprovado = None
    melhor_aic_aprovado = np.inf

    total = len(ordens) * len(sazonais)
    contador = 0
    for ordem in ordens:
        for ordem_sazonal in sazonais:
            contador += 1
            inicio = time.perf_counter()
            print(f"[{contador:02d}/{total}] SARIMA{ordem}x{ordem_sazonal}...", flush=True)
            registro = {
                "order": str(ordem),
                "seasonal_order": str(ordem_sazonal),
                "AIC": np.nan,
                "BIC": np.nan,
                "Convergiu": False,
                "LjungBox_p_12": np.nan,
                "LjungBox_p_24": np.nan,
                "Residuos_OK_5pct": False,
                "Tempo_segundos": np.nan,
                "Status": "",
            }
            try:
                with warnings.catch_warnings(record=True) as avisos:
                    warnings.simplefilter("always")
                    modelo = SARIMAX(
                        treino,
                        order=ordem,
                        seasonal_order=ordem_sazonal,
                        trend="n",
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    ajuste = modelo.fit(disp=False, maxiter=300, method="lbfgs")
                convergiu = bool(ajuste.mle_retvals.get("converged", False))
                aic = float(ajuste.aic)
                bic = float(ajuste.bic)
                finitos = np.isfinite(aic) and np.isfinite(bic)
                diagnostico = diagnostico_residuos(ajuste, ordem, ordem_sazonal)
                p12 = float(diagnostico.loc[diagnostico["Lag"] == 12, "lb_pvalue"].iloc[0])
                p24 = float(diagnostico.loc[diagnostico["Lag"] == 24, "lb_pvalue"].iloc[0])
                residuos_ok = bool(np.isfinite(p12) and np.isfinite(p24) and p12 >= 0.05 and p24 >= 0.05)
                registro.update(
                    {
                        "AIC": aic,
                        "BIC": bic,
                        "Convergiu": convergiu,
                        "LjungBox_p_12": p12,
                        "LjungBox_p_24": p24,
                        "Residuos_OK_5pct": residuos_ok,
                        "Status": "OK" if convergiu and finitos else "DESCARTADO",
                        "Avisos": " | ".join(str(a.message) for a in avisos),
                    }
                )
                if convergiu and finitos and aic < melhor_aic_geral:
                    melhor_aic_geral = aic
                    melhor_geral = ajuste
                if convergiu and finitos and residuos_ok and aic < melhor_aic_aprovado:
                    melhor_aic_aprovado = aic
                    melhor_aprovado = ajuste
            except Exception as exc:
                registro["Status"] = f"ERRO: {type(exc).__name__}: {exc}"
            registro["Tempo_segundos"] = time.perf_counter() - inicio
            resultados.append(registro)

    tabela = pd.DataFrame(resultados).sort_values(
        ["Residuos_OK_5pct", "Convergiu", "AIC"],
        ascending=[False, False, True],
        na_position="last",
    )
    if melhor_geral is None:
        raise RuntimeError("Nenhum SARIMA convergiu com AIC finito.")
    if melhor_aprovado is not None:
        escolhido = melhor_aprovado
        regra = "Menor AIC entre modelos convergentes com Ljung-Box p>=0,05 nos lags 12 e 24"
        aprovou = True
    else:
        escolhido = melhor_geral
        regra = "Fallback: menor AIC geral; nenhum candidato aprovou simultaneamente os dois Ljung-Box"
        aprovou = False
    metadados = {
        "regra_selecao": regra,
        "residuos_aprovados": aprovou,
        "quantidade_convergente": int(tabela["Convergiu"].sum()),
        "quantidade_residuos_ok": int(tabela["Residuos_OK_5pct"].sum()),
        "menor_aic_geral": float(melhor_aic_geral),
    }
    return escolhido, tabela, metadados


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa SARIMA e compara tres modelos.")
    parser.add_argument("base", nargs="?", help="Caminho opcional da planilha oficial filtrada.")
    args = parser.parse_args()

    base = localizar_base(args.base)
    pasta_saida = Path(__file__).resolve().parent / "resultados_sarima_ampliado"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    print(f"Base: {base}")
    serie = carregar_serie(base)
    treino = serie.iloc[:-N_TESTE]
    teste = serie.iloc[-N_TESTE:]
    print(f"Treino: {treino.index.min():%Y-%m} a {treino.index.max():%Y-%m} ({len(treino)} meses)")
    print(f"Teste:  {teste.index.min():%Y-%m} a {teste.index.max():%Y-%m} ({len(teste)} meses)")

    # 1) Benchmark ingenuo sazonal.
    prev_ingenuo = pd.Series(serie.shift(SAZONALIDADE).loc[teste.index], index=teste.index)

    # 2) Holt-Winters reestimado com a biblioteca oficial para validacao cruzada.
    ajuste_hw, candidatos_hw = ajustar_holt_winters(treino)
    prev_hw = pd.Series(ajuste_hw.forecast(N_TESTE), index=teste.index)
    hw_escolhido = candidatos_hw.loc[candidatos_hw["SSE_treinamento"].idxmin(), "Modelo"]

    # 3) SARIMA escolhido pelo AIC do treinamento, sem olhar o teste.
    ajuste_sarima, busca, selecao_sarima = busca_sarima(treino)
    objeto_prev = ajuste_sarima.get_forecast(steps=N_TESTE)
    prev_sarima = pd.Series(objeto_prev.predicted_mean.to_numpy(), index=teste.index)
    intervalo = objeto_prev.conf_int(alpha=0.05)

    metricas_modelos = {
        "Ingenio sazonal": metricas(teste, prev_ingenuo),
        "Holt-Winters": metricas(teste, prev_hw),
        "SARIMA": metricas(teste, prev_sarima),
    }
    comparacao = pd.DataFrame(metricas_modelos).T.reset_index(names="Modelo")
    comparacao = comparacao.sort_values(["MAPE_percentual", "RMSE_GWh", "MAE_GWh"])
    comparacao.insert(1, "Posicao_MAPE", range(1, len(comparacao) + 1))

    ordem = tuple(int(x) for x in ajuste_sarima.model.order)
    ordem_sazonal = tuple(int(x) for x in ajuste_sarima.model.seasonal_order)
    modelo_sarima = f"SARIMA{ordem}x{ordem_sazonal}"

    ljung = diagnostico_residuos(ajuste_sarima, ordem, ordem_sazonal)

    detalhe = pd.DataFrame(
        {
            "Data": teste.index,
            "Real_GWh": teste.to_numpy(),
            "Previsao_Ingenio_GWh": prev_ingenuo.to_numpy(),
            "Previsao_Holt_Winters_GWh": prev_hw.to_numpy(),
            "Previsao_SARIMA_GWh": prev_sarima.to_numpy(),
            "SARIMA_Limite_Inferior_95_GWh": intervalo.iloc[:, 0].to_numpy(),
            "SARIMA_Limite_Superior_95_GWh": intervalo.iloc[:, 1].to_numpy(),
        }
    )
    detalhe["Erro_SARIMA_GWh"] = detalhe["Real_GWh"] - detalhe["Previsao_SARIMA_GWh"]
    detalhe["APE_SARIMA_percentual"] = (
        detalhe["Erro_SARIMA_GWh"].abs() / detalhe["Real_GWh"] * 100
    )

    # Arquivos simples e auditaveis; a planilha acadêmica final sera montada apos a revisao.
    busca.to_csv(pasta_saida / "01_busca_sarima.csv", index=False, encoding="utf-8-sig")
    candidatos_hw.to_csv(pasta_saida / "02_busca_holt_winters.csv", index=False, encoding="utf-8-sig")
    comparacao.to_csv(pasta_saida / "03_comparacao_modelos.csv", index=False, encoding="utf-8-sig")
    detalhe.to_csv(pasta_saida / "04_resultados_teste.csv", index=False, encoding="utf-8-sig")
    ljung.to_csv(pasta_saida / "05_diagnostico_ljung_box.csv", index=False, encoding="utf-8-sig")

    resumo = {
        "base": str(base),
        "periodo_total": [str(serie.index.min().date()), str(serie.index.max().date())],
        "n_total": len(serie),
        "periodo_treino": [str(treino.index.min().date()), str(treino.index.max().date())],
        "n_treino": len(treino),
        "periodo_teste": [str(teste.index.min().date()), str(teste.index.max().date())],
        "n_teste": len(teste),
        "criterio_selecao_sarima": selecao_sarima["regra_selecao"],
        "grade_sarima": "p,q,P,Q em {0,1,2}; d=1; D=1; s=12; 81 candidatos",
        "quantidade_modelos_convergentes": selecao_sarima["quantidade_convergente"],
        "quantidade_modelos_com_residuos_ok": selecao_sarima["quantidade_residuos_ok"],
        "modelo_escolhido_aprovou_residuos": selecao_sarima["residuos_aprovados"],
        "menor_aic_geral": selecao_sarima["menor_aic_geral"],
        "sarima_escolhido": modelo_sarima,
        "sarima_aic": float(ajuste_sarima.aic),
        "sarima_bic": float(ajuste_sarima.bic),
        "holt_winters_escolhido": hw_escolhido,
        "metricas": metricas_modelos,
        "melhor_modelo_no_teste_por_mape": str(comparacao.iloc[0]["Modelo"]),
        "ambiente": {
            "python": sys.version,
            "plataforma": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (pasta_saida / "06_resumo_execucao.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(12, 6.5))
    plt.plot(treino.iloc[-36:].index, treino.iloc[-36:], color="#6B7280", label="Treino (ultimos 36 meses)")
    plt.plot(teste.index, teste, color="#111827", linewidth=2.4, label="Real - teste")
    plt.plot(teste.index, prev_ingenuo, color="#D97706", linestyle="--", label="Ingenuo sazonal")
    plt.plot(teste.index, prev_hw, color="#2563EB", linestyle="--", label="Holt-Winters")
    plt.plot(teste.index, prev_sarima, color="#059669", linestyle="--", linewidth=2, label=modelo_sarima)
    plt.fill_between(
        teste.index,
        intervalo.iloc[:, 0].to_numpy(),
        intervalo.iloc[:, 1].to_numpy(),
        color="#A7F3D0",
        alpha=0.35,
        label="Intervalo SARIMA 95%",
    )
    plt.axvline(teste.index.min(), color="#9CA3AF", linestyle=":", linewidth=1)
    plt.title("Consumo residencial: previsoes no periodo de teste")
    plt.ylabel("Consumo (GWh)")
    plt.xlabel("Mes")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(pasta_saida / "07_grafico_comparacao.png", dpi=180)
    plt.close()

    print("\nCOMPARACAO NO TESTE")
    print(comparacao.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nSARIMA escolhido pelo AIC: {modelo_sarima}")
    print(f"Resultados salvos em: {pasta_saida}")


if __name__ == "__main__":
    main()
