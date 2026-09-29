r"""Modelo 04 do TCC: validacao por origem movel dos tres metodos.

Uso no PowerShell (arquivo da base na mesma pasta):
    python modelo_04_validacao_origem_movel.py

Ou informando o caminho da base:
    python modelo_04_validacao_origem_movel.py "C:\\caminho\\Base_Oficial_....xlsx"

O script executa cinco janelas anuais, de jul/2021-jun/2022 ate
jul/2025-jun/2026. Em cada janela, os modelos sao reajustados apenas com os dados
anteriores. A busca de 81 SARIMA tambem e repetida dentro de cada treinamento,
evitando vazamento temporal.
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


def busca_sarima(treino: pd.Series, prefixo: str = "") -> tuple[object, pd.DataFrame, dict]:
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
            print(f"{prefixo}[{contador:02d}/{total}] SARIMA{ordem}x{ordem_sazonal}...", flush=True)
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
    parser = argparse.ArgumentParser(description="Validacao por origem movel de tres modelos.")
    parser.add_argument("base", nargs="?", help="Caminho opcional da planilha oficial filtrada.")
    args = parser.parse_args()

    base = localizar_base(args.base)
    pasta_saida = Path(__file__).resolve().parent / "resultados_origem_movel"
    pasta_saida.mkdir(parents=True, exist_ok=True)

    print(f"Base: {base}")
    serie = carregar_serie(base)
    inicios_teste = [pd.Timestamp(ano, 7, 1) for ano in range(2021, 2026)]
    metricas_janelas = []
    previsoes_janelas = []
    buscas_janelas = []
    modelos_janelas = []

    for numero, inicio_teste in enumerate(inicios_teste, start=1):
        fim_teste = inicio_teste + pd.DateOffset(months=N_TESTE - 1)
        treino = serie.loc[serie.index < inicio_teste]
        teste = serie.loc[inicio_teste:fim_teste]
        if len(teste) != N_TESTE:
            raise ValueError(f"Janela {inicio_teste:%Y-%m} nao possui 12 meses completos.")

        janela = f"{inicio_teste:%Y-%m}_a_{fim_teste:%Y-%m}"
        print("\n" + "=" * 72)
        print(f"JANELA {numero}/5: {janela}")
        print(f"Treino: {treino.index.min():%Y-%m} a {treino.index.max():%Y-%m} ({len(treino)} meses)")
        print(f"Teste:  {teste.index.min():%Y-%m} a {teste.index.max():%Y-%m} ({len(teste)} meses)")

        prev_ingenuo = pd.Series(serie.shift(SAZONALIDADE).loc[teste.index], index=teste.index)

        ajuste_hw, candidatos_hw = ajustar_holt_winters(treino)
        prev_hw = pd.Series(ajuste_hw.forecast(N_TESTE), index=teste.index)
        hw_escolhido = candidatos_hw.loc[candidatos_hw["SSE_treinamento"].idxmin(), "Modelo"]

        prefixo = f"[Janela {numero}/5]"
        ajuste_sarima, busca, selecao = busca_sarima(treino, prefixo=prefixo)
        prev_sarima = pd.Series(
            ajuste_sarima.get_forecast(steps=N_TESTE).predicted_mean.to_numpy(),
            index=teste.index,
        )

        ordem = tuple(int(x) for x in ajuste_sarima.model.order)
        ordem_sazonal = tuple(int(x) for x in ajuste_sarima.model.seasonal_order)
        nome_sarima = f"SARIMA{ordem}x{ordem_sazonal}"
        ljung = diagnostico_residuos(ajuste_sarima, ordem, ordem_sazonal)
        p12 = float(ljung.loc[ljung["Lag"] == 12, "lb_pvalue"].iloc[0])
        p24 = float(ljung.loc[ljung["Lag"] == 24, "lb_pvalue"].iloc[0])

        busca.insert(0, "Janela", janela)
        buscas_janelas.append(busca)
        modelos_janelas.append(
            {
                "Janela": janela,
                "Treino_inicio": treino.index.min(),
                "Treino_fim": treino.index.max(),
                "N_treino": len(treino),
                "SARIMA_escolhido": nome_sarima,
                "AIC": float(ajuste_sarima.aic),
                "BIC": float(ajuste_sarima.bic),
                "LjungBox_p_12": p12,
                "LjungBox_p_24": p24,
                "Residuos_OK_5pct": selecao["residuos_aprovados"],
                "Candidatos_convergentes": selecao["quantidade_convergente"],
                "Candidatos_residuos_OK": selecao["quantidade_residuos_ok"],
                "Holt_Winters_escolhido": hw_escolhido,
                "Regra_selecao_SARIMA": selecao["regra_selecao"],
            }
        )

        previsoes = {
            "Ingenio sazonal": prev_ingenuo,
            "Holt-Winters": prev_hw,
            "SARIMA": prev_sarima,
        }
        for modelo, previsao in previsoes.items():
            valores = metricas(teste, previsao)
            metricas_janelas.append({"Janela": janela, "Modelo": modelo, **valores})
            for data, real, previsto in zip(teste.index, teste.to_numpy(), previsao.to_numpy()):
                previsoes_janelas.append(
                    {
                        "Janela": janela,
                        "Data": data,
                        "Modelo": modelo,
                        "Real_GWh": float(real),
                        "Previsao_GWh": float(previsto),
                        "Erro_GWh": float(real - previsto),
                        "APE_percentual": float(abs(real - previsto) / real * 100),
                    }
                )

    df_metricas = pd.DataFrame(metricas_janelas)
    df_previsoes = pd.DataFrame(previsoes_janelas)
    df_buscas = pd.concat(buscas_janelas, ignore_index=True)
    df_modelos = pd.DataFrame(modelos_janelas)

    linhas_agregadas = []
    for modelo, grupo in df_previsoes.groupby("Modelo", sort=False):
        real = pd.Series(grupo["Real_GWh"].to_numpy())
        previsto = pd.Series(grupo["Previsao_GWh"].to_numpy())
        agregado = metricas(real, previsto)
        mape_janelas = df_metricas.loc[df_metricas["Modelo"] == modelo, "MAPE_percentual"]
        agregado.update(
            {
                "Modelo": modelo,
                "MAPE_medio_das_janelas": float(mape_janelas.mean()),
                "Desvio_padrao_MAPE_janelas": float(mape_janelas.std(ddof=1)),
            }
        )
        linhas_agregadas.append(agregado)

    df_agregado = pd.DataFrame(linhas_agregadas)
    vencedores = df_metricas.loc[df_metricas.groupby("Janela")["MAPE_percentual"].idxmin()]
    contagem_vitorias = vencedores["Modelo"].value_counts()
    df_agregado["Vitorias_MAPE_em_5_janelas"] = (
        df_agregado["Modelo"].map(contagem_vitorias).fillna(0).astype(int)
    )
    df_agregado = df_agregado.sort_values(
        ["MAPE_percentual", "RMSE_GWh", "MAE_GWh"]
    ).reset_index(drop=True)
    df_agregado.insert(1, "Posicao_MAPE_60_meses", range(1, len(df_agregado) + 1))
    vencedor = str(df_agregado.iloc[0]["Modelo"])

    df_buscas.to_csv(pasta_saida / "01_busca_sarima_por_janela.csv", index=False, encoding="utf-8-sig")
    df_metricas.to_csv(pasta_saida / "02_metricas_por_janela.csv", index=False, encoding="utf-8-sig")
    df_agregado.to_csv(pasta_saida / "03_metricas_agregadas_60_meses.csv", index=False, encoding="utf-8-sig")
    df_previsoes.to_csv(pasta_saida / "04_previsoes_origem_movel.csv", index=False, encoding="utf-8-sig")
    df_modelos.to_csv(pasta_saida / "05_modelos_selecionados.csv", index=False, encoding="utf-8-sig")

    resumo = {
        "base": str(base),
        "periodo_total": [str(serie.index.min().date()), str(serie.index.max().date())],
        "metodo": "Origem movel expansiva com cinco testes anuais de 12 meses",
        "janelas": [f"{x:%Y-%m}_a_{(x + pd.DateOffset(months=11)):%Y-%m}" for x in inicios_teste],
        "grade_sarima_por_janela": "p,q,P,Q em {0,1,2}; d=1; D=1; s=12; 81 candidatos",
        "total_ajustes_sarima_planejados": 405,
        "criterio_principal": "Menor MAPE agregado nas 60 previsoes; RMSE e MAE como desempate",
        "vencedor_mape_60_meses": vencedor,
        "metricas_agregadas": json.loads(df_agregado.to_json(orient="records")),
        "ambiente": {
            "python": sys.version,
            "plataforma": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (pasta_saida / "06_resumo_validacao.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cores = {"Ingenio sazonal": "#D97706", "Holt-Winters": "#2563EB", "SARIMA": "#059669"}
    pivot_mape = df_metricas.pivot(index="Janela", columns="Modelo", values="MAPE_percentual")
    ax = pivot_mape.plot(kind="bar", figsize=(12, 6), color=[cores.get(c, "#6B7280") for c in pivot_mape.columns])
    ax.set_title("MAPE por janela anual na validacao por origem movel")
    ax.set_xlabel("Janela de teste")
    ax.set_ylabel("MAPE (%)")
    ax.grid(axis="y", alpha=0.22)
    ax.legend(frameon=False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(pasta_saida / "07_grafico_mape_por_janela.png", dpi=180)
    plt.close()

    plt.figure(figsize=(13, 6.5))
    real_unico = df_previsoes.drop_duplicates("Data").sort_values("Data")
    plt.plot(real_unico["Data"], real_unico["Real_GWh"], color="#111827", linewidth=2.2, label="Real")
    for modelo, grupo in df_previsoes.groupby("Modelo"):
        grupo = grupo.sort_values("Data")
        plt.plot(grupo["Data"], grupo["Previsao_GWh"], linestyle="--", color=cores[modelo], label=modelo)
    plt.title("Previsoes nas cinco janelas da validacao por origem movel")
    plt.xlabel("Mes")
    plt.ylabel("Consumo (GWh)")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(pasta_saida / "08_grafico_previsoes_60_meses.png", dpi=180)
    plt.close()

    print("\nMETRICAS AGREGADAS NAS 60 PREVISOES")
    print(df_agregado.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nVencedor pelo MAPE agregado: {vencedor}")
    print(f"Resultados salvos em: {pasta_saida}")


if __name__ == "__main__":
    main()
