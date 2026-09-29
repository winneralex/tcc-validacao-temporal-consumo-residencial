r"""Modelo final do TCC: Holt-Winters e previsao de jul/2026 a jun/2027.

Uso no PowerShell (arquivo da base na mesma pasta):
    python modelo_05_previsao_final_holt_winters_bootstrap.py

Ou informando o caminho da base:
    python modelo_05_previsao_final_holt_winters_bootstrap.py "C:\\caminho\\Base_Oficial_....xlsx"

O modelo usa tendencia aditiva e sazonalidade multiplicativa com periodo 12,
configuracao vencedora na validacao por origem movel. Os intervalos de 95% sao
estimados por 10.000 simulacoes com bootstrap em blocos de 12 meses dos residuos
relativos, preservando a distribuicao empirica e a dependencia dentro de cada bloco.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import statsmodels
from scipy.stats import jarque_bera
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.holtwinters import ExponentialSmoothing


HORIZONTE = 12
SAZONALIDADE = 12
REPETICOES = 10_000
SEMENTE = 20260825


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

    esperado = pd.date_range(dados["Data"].min(), dados["Data"].max(), freq="MS")
    if not dados["Data"].reset_index(drop=True).equals(pd.Series(esperado)):
        raise ValueError("A serie nao e mensal completa.")

    serie = pd.Series(
        dados["Consumo_GWh"].to_numpy(dtype=float),
        index=pd.DatetimeIndex(dados["Data"], freq="MS"),
        name="Consumo_GWh",
    )
    if len(serie) != 270 or serie.index.min() != pd.Timestamp("2004-01-01") or serie.index.max() != pd.Timestamp("2026-06-01"):
        raise ValueError("A base nao corresponde ao recorte aprovado de 270 meses.")
    return serie


def main() -> None:
    parser = argparse.ArgumentParser(description="Reajusta Holt-Winters e gera a previsao final.")
    parser.add_argument("base", nargs="?", help="Caminho opcional da planilha oficial filtrada.")
    args = parser.parse_args()

    base = localizar_base(args.base)
    pasta_saida = Path(__file__).resolve().parent / "resultados_previsao_final_bootstrap"
    pasta_saida.mkdir(parents=True, exist_ok=True)
    serie = carregar_serie(base)

    print(f"Base: {base}")
    print(f"Serie completa: {serie.index.min():%Y-%m} a {serie.index.max():%Y-%m} ({len(serie)} meses)")
    print("Ajustando Holt-Winters final...")

    modelo = ExponentialSmoothing(
        serie,
        trend="add",
        seasonal="mul",
        seasonal_periods=SAZONALIDADE,
        damped_trend=False,
        initialization_method="estimated",
    )
    ajuste = modelo.fit(optimized=True, remove_bias=False)

    futuro = pd.date_range(serie.index.max() + pd.offsets.MonthBegin(1), periods=HORIZONTE, freq="MS")
    previsao = pd.Series(ajuste.forecast(HORIZONTE).to_numpy(dtype=float), index=futuro)

    # Bootstrap em blocos contiguos de 12 meses. Os residuos relativos sao
    # adequados ao erro multiplicativo e sao centrados antes da reamostragem.
    ajustados = pd.Series(ajuste.fittedvalues, index=serie.index)
    residuos_relativos = ((serie - ajustados) / ajustados).iloc[2 * SAZONALIDADE :]
    residuos_relativos = residuos_relativos[np.isfinite(residuos_relativos)]
    residuos_relativos = residuos_relativos - residuos_relativos.mean()
    if len(residuos_relativos) < HORIZONTE:
        raise RuntimeError("Nao ha residuos suficientes para o bootstrap em blocos.")

    gerador = np.random.default_rng(SEMENTE)
    valores_residuais = residuos_relativos.to_numpy(dtype=float)
    inicios_blocos = gerador.integers(
        0,
        len(valores_residuais) - HORIZONTE + 1,
        size=REPETICOES,
    )
    indices_blocos = np.arange(HORIZONTE)[:, None] + inicios_blocos[None, :]
    erros_bootstrap = valores_residuais[indices_blocos]

    print(f"Gerando {REPETICOES:,} trajetorias com bootstrap em blocos para os intervalos de 95%...")
    simulacoes = ajuste.simulate(
        HORIZONTE,
        anchor="end",
        repetitions=REPETICOES,
        error="mul",
        random_errors=erros_bootstrap,
    )
    matriz = np.asarray(simulacoes, dtype=float)
    if matriz.shape == (REPETICOES, HORIZONTE):
        matriz = matriz.T
    if matriz.shape != (HORIZONTE, REPETICOES):
        raise RuntimeError(f"Formato inesperado das simulacoes: {matriz.shape}")
    if not np.isfinite(matriz).all():
        raise RuntimeError("As simulacoes produziram valores nao finitos.")

    limite_inferior = np.quantile(matriz, 0.025, axis=1)
    mediana_simulada = np.quantile(matriz, 0.500, axis=1)
    limite_superior = np.quantile(matriz, 0.975, axis=1)
    referencia = serie.loc[futuro - pd.DateOffset(years=1)].to_numpy(dtype=float)

    tabela = pd.DataFrame(
        {
            "Data": futuro,
            "Previsao_GWh": previsao.to_numpy(),
            "Limite_Inferior_95_GWh": limite_inferior,
            "Mediana_Simulada_GWh": mediana_simulada,
            "Limite_Superior_95_GWh": limite_superior,
            "Real_Mes_Ano_Anterior_GWh": referencia,
        }
    )
    tabela["Variacao_vs_Ano_Anterior_percentual"] = (
        tabela["Previsao_GWh"] / tabela["Real_Mes_Ano_Anterior_GWh"] - 1
    ) * 100
    tabela["Amplitude_Intervalo_95_GWh"] = (
        tabela["Limite_Superior_95_GWh"] - tabela["Limite_Inferior_95_GWh"]
    )

    residuos = pd.Series(ajuste.resid, index=serie.index, name="Residuo_GWh")
    residuos_validos = residuos.iloc[2 * SAZONALIDADE :]
    ljung = acorr_ljungbox(residuos_validos, lags=[12, 24], return_df=True).reset_index(names="Lag")
    jb = jarque_bera(residuos_validos.to_numpy())
    diagnosticos = pd.DataFrame(
        [
            {"Indicador": "N_residuos_diagnostico", "Valor": float(len(residuos_validos))},
            {"Indicador": "Media_residuos_GWh", "Valor": float(residuos_validos.mean())},
            {"Indicador": "Desvio_padrao_residuos_GWh", "Valor": float(residuos_validos.std(ddof=1))},
            {"Indicador": "RMSE_ajuste_GWh", "Valor": float(np.sqrt(np.mean(residuos_validos**2)))},
            {"Indicador": "MAPE_ajuste_percentual", "Valor": float(np.mean(np.abs(residuos_validos / serie.loc[residuos_validos.index])) * 100)},
            {"Indicador": "Autocorrelacao_lag_1", "Valor": float(residuos_validos.autocorr(lag=1))},
            {"Indicador": "LjungBox_p_12", "Valor": float(ljung.loc[ljung["Lag"] == 12, "lb_pvalue"].iloc[0])},
            {"Indicador": "LjungBox_p_24", "Valor": float(ljung.loc[ljung["Lag"] == 24, "lb_pvalue"].iloc[0])},
            {"Indicador": "Jarque_Bera_p", "Valor": float(jb.pvalue)},
        ]
    )

    parametros = ajuste.params_formatted.reset_index().rename(columns={"index": "Parametro"})
    total_previsto = float(tabela["Previsao_GWh"].sum())
    total_referencia = float(tabela["Real_Mes_Ano_Anterior_GWh"].sum())
    variacao_total = (total_previsto / total_referencia - 1) * 100

    tabela.to_csv(pasta_saida / "01_previsao_final_holt_winters.csv", index=False, encoding="utf-8-sig")
    parametros.to_csv(pasta_saida / "02_parametros_modelo_final.csv", index=False, encoding="utf-8-sig")
    diagnosticos.to_csv(pasta_saida / "03_diagnosticos_modelo_final.csv", index=False, encoding="utf-8-sig")
    residuos.rename_axis("Data").reset_index().to_csv(
        pasta_saida / "04_residuos_historicos.csv", index=False, encoding="utf-8-sig"
    )

    resumo = {
        "base": str(base),
        "periodo_treinamento_final": [str(serie.index.min().date()), str(serie.index.max().date())],
        "n_treinamento": len(serie),
        "modelo_final": "Holt-Winters com tendencia aditiva e sazonalidade multiplicativa",
        "periodo_sazonal": SAZONALIDADE,
        "periodo_previsao": [str(futuro.min().date()), str(futuro.max().date())],
        "horizonte_meses": HORIZONTE,
        "intervalo": {
            "nivel": 0.95,
            "metodo": "Bootstrap em blocos contiguos de 12 meses dos residuos relativos centrados, com erro multiplicativo",
            "repeticoes": REPETICOES,
            "semente": SEMENTE,
            "tamanho_bloco_meses": HORIZONTE,
            "n_residuos_disponiveis": int(len(residuos_relativos)),
            "media_residuos_relativos_apos_centragem": float(residuos_relativos.mean()),
        },
        "sse": float(ajuste.sse),
        "aic": float(ajuste.aic),
        "aicc": float(ajuste.aicc),
        "bic": float(ajuste.bic),
        "total_previsto_jul2026_jun2027_GWh": total_previsto,
        "total_real_jul2025_jun2026_GWh": total_referencia,
        "variacao_total_prevista_percentual": float(variacao_total),
        "ambiente": {
            "python": sys.version,
            "plataforma": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "statsmodels": statsmodels.__version__,
        },
    }
    (pasta_saida / "05_resumo_previsao_final.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plt.figure(figsize=(13, 6.8))
    historico = serie.iloc[-60:]
    plt.plot(historico.index, historico, color="#111827", linewidth=2.1, label="Historico")
    plt.plot(futuro, previsao, color="#2563EB", linewidth=2.3, linestyle="--", label="Previsao Holt-Winters")
    plt.fill_between(
        futuro,
        limite_inferior,
        limite_superior,
        color="#93C5FD",
        alpha=0.35,
        label="Intervalo de previsao de 95%",
    )
    plt.axvline(futuro.min(), color="#6B7280", linestyle=":", linewidth=1.2)
    plt.title("Previsao final do consumo residencial de energia eletrica no Brasil")
    plt.xlabel("Mes")
    plt.ylabel("Consumo (GWh)")
    plt.grid(axis="y", alpha=0.22)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(pasta_saida / "06_grafico_previsao_final.png", dpi=180)
    plt.close()

    print("\nPREVISAO FINAL")
    print(
        tabela[["Data", "Previsao_GWh", "Limite_Inferior_95_GWh", "Limite_Superior_95_GWh"]]
        .to_string(index=False, float_format=lambda x: f"{x:.3f}")
    )
    print(f"\nTotal previsto: {total_previsto:,.3f} GWh")
    print(f"Variacao frente aos 12 meses anteriores: {variacao_total:.3f}%")
    print(f"Resultados salvos em: {pasta_saida}")


if __name__ == "__main__":
    main()
