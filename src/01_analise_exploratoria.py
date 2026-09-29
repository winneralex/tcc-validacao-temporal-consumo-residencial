#!/usr/bin/env python3
"""Análise exploratória reproduzível do consumo residencial de energia no Brasil.

Uso:
    python analise_exploratoria_energia.py BASE.xlsx saida.json

A planilha de entrada deve conter a aba ``Residencial Brasil`` produzida a
partir dos dados abertos da Empresa de Pesquisa Energética (EPE).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


MESES_PT = {
    1: "Janeiro",
    2: "Fevereiro",
    3: "Março",
    4: "Abril",
    5: "Maio",
    6: "Junho",
    7: "Julho",
    8: "Agosto",
    9: "Setembro",
    10: "Outubro",
    11: "Novembro",
    12: "Dezembro",
}


def _json_value(value):
    if value is None or value is pd.NA:
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def records(df: pd.DataFrame) -> list[dict]:
    return [
        {str(column): _json_value(value) for column, value in row.items()}
        for row in df.to_dict(orient="records")
    ]


def analyse(input_path: Path) -> dict:
    source = pd.read_excel(input_path, sheet_name="Residencial Brasil")
    required = {"Data", "Consumo_MWh", "Consumidores"}
    missing_columns = required.difference(source.columns)
    if missing_columns:
        raise ValueError(f"Colunas obrigatórias ausentes: {sorted(missing_columns)}")

    df = source.loc[:, ["Data", "Consumo_MWh", "Consumidores"]].copy()
    df["Data"] = pd.to_datetime(df["Data"], errors="raise")
    df = df.sort_values("Data").reset_index(drop=True)
    df["Consumo_MWh"] = pd.to_numeric(df["Consumo_MWh"], errors="raise")
    df["Consumidores"] = pd.to_numeric(df["Consumidores"], errors="coerce")
    df["Consumo_GWh"] = df["Consumo_MWh"] / 1000.0

    expected_dates = pd.date_range(df["Data"].min(), df["Data"].max(), freq="MS")
    missing_dates = expected_dates.difference(pd.DatetimeIndex(df["Data"]))
    duplicate_dates = int(df["Data"].duplicated().sum())
    invalid_consumption = int((~np.isfinite(df["Consumo_GWh"]) | (df["Consumo_GWh"] <= 0)).sum())

    if duplicate_dates or len(missing_dates) or invalid_consumption:
        raise ValueError(
            "A série não passou nos controles essenciais: "
            f"duplicidades={duplicate_dates}, meses_ausentes={len(missing_dates)}, "
            f"consumos_invalidos={invalid_consumption}"
        )

    df["Ano"] = df["Data"].dt.year
    df["Mes_Num"] = df["Data"].dt.month
    df["Mes"] = df["Mes_Num"].map(MESES_PT)
    df["Media_Movel_12m_GWh"] = df["Consumo_GWh"].rolling(12).mean()
    df["Total_Movel_12m_GWh"] = df["Consumo_GWh"].rolling(12).sum()
    df["Variacao_Mensal_pct"] = df["Consumo_GWh"].pct_change() * 100
    df["Variacao_Anual_pct"] = df["Consumo_GWh"].pct_change(12) * 100

    # Decomposição aditiva clássica com periodicidade anual. Para um período
    # par (12 meses), a tendência é obtida por uma média móvel 2 x 12 e
    # centralizada. O componente sazonal é a média mensal da série sem tendência.
    indexed = df.set_index("Data")["Consumo_GWh"]
    trend = indexed.rolling(12).mean().rolling(2).mean().shift(-6)
    detrended = indexed - trend
    seasonal_pattern = detrended.groupby(detrended.index.month).mean()
    seasonal_pattern = seasonal_pattern - seasonal_pattern.mean()
    seasonal = pd.Series(
        indexed.index.month.map(seasonal_pattern).to_numpy(),
        index=indexed.index,
        dtype=float,
    )
    residual = indexed - trend - seasonal
    decomposition = pd.DataFrame(
        {
            "Data": df["Data"],
            "Observado_GWh": df["Consumo_GWh"],
            "Tendencia_GWh": trend.to_numpy(),
            "Sazonal_GWh": seasonal.to_numpy(),
            "Residuo_GWh": residual.to_numpy(),
        }
    )

    resid = decomposition["Residuo_GWh"].dropna()
    median_resid = float(resid.median())
    mad = float(np.median(np.abs(resid - median_resid)))
    if mad > 0:
        robust_z_valid = 0.6744897501960817 * (resid - median_resid) / mad
        outlier_method = "Resíduo da decomposição aditiva clássica (período 12) com |z robusto| ≥ 3,5"
    else:
        std = float(resid.std(ddof=1))
        robust_z_valid = (resid - float(resid.mean())) / std if std > 0 else pd.Series(0.0, index=resid.index)
        outlier_method = "Resíduo da decomposição aditiva clássica (período 12) com |z| ≥ 3,5"
    robust_z = pd.Series(np.nan, index=decomposition.index, dtype=float)
    robust_z.loc[resid.index] = robust_z_valid.to_numpy()
    decomposition["Z_Robusto"] = robust_z
    decomposition["Periodo_Atipico"] = np.where(
        robust_z.isna(), None, robust_z.abs() >= 3.5
    )

    outlier_mask = decomposition["Periodo_Atipico"].eq(True)
    outliers = decomposition.loc[outlier_mask].copy()
    outliers = outliers.merge(
        df.loc[:, ["Data", "Consumo_GWh", "Variacao_Anual_pct"]], on="Data", how="left"
    )
    outliers["Direcao_Residuo"] = np.where(
        outliers["Residuo_GWh"] >= 0, "Acima do esperado", "Abaixo do esperado"
    )
    outliers = outliers.sort_values("Data")

    annual = (
        df.groupby("Ano", as_index=False)
        .agg(
            Meses=("Data", "count"),
            Total_GWh=("Consumo_GWh", "sum"),
            Media_Mensal_GWh=("Consumo_GWh", "mean"),
            Minimo_Mensal_GWh=("Consumo_GWh", "min"),
            Maximo_Mensal_GWh=("Consumo_GWh", "max"),
        )
    )
    annual["Situacao"] = np.where(annual["Meses"] == 12, "Completo", "Parcial")
    annual["Variacao_Anual_pct"] = annual["Total_GWh"].pct_change() * 100
    annual.loc[
        (annual["Meses"] != 12) | (annual["Meses"].shift(1) != 12),
        "Variacao_Anual_pct",
    ] = np.nan

    complete_years = (
        df.groupby("Ano")["Data"].count().loc[lambda s: s.eq(12)].index
    )
    seasonal_source = df.loc[df["Ano"].isin(complete_years)].copy()
    seasonal_source["Media_Ano_GWh"] = seasonal_source.groupby("Ano")["Consumo_GWh"].transform("mean")
    seasonal_source["Indice_Dentro_Ano"] = seasonal_source["Consumo_GWh"] / seasonal_source["Media_Ano_GWh"] * 100
    seasonality = (
        seasonal_source.groupby(["Mes_Num", "Mes"], as_index=False)
        .agg(
            Observacoes=("Consumo_GWh", "count"),
            Media_GWh=("Consumo_GWh", "mean"),
            Mediana_GWh=("Consumo_GWh", "median"),
            Desvio_Padrao_GWh=("Consumo_GWh", "std"),
            Minimo_GWh=("Consumo_GWh", "min"),
            Maximo_GWh=("Consumo_GWh", "max"),
            Indice_Sazonal_Medio=("Indice_Dentro_Ano", "mean"),
        )
        .sort_values("Mes_Num")
    )

    complete_annual = annual.loc[annual["Situacao"] == "Completo"].copy()
    first_complete = complete_annual.iloc[0]
    last_complete = complete_annual.iloc[-1]
    annual_change = (
        last_complete["Total_GWh"] / first_complete["Total_GWh"] - 1
    ) * 100
    annual_change_pt = f"{annual_change:.2f}".replace(".", ",")
    high_month = seasonality.loc[seasonality["Indice_Sazonal_Medio"].idxmax()]
    low_month = seasonality.loc[seasonality["Indice_Sazonal_Medio"].idxmin()]
    min_idx = df["Consumo_GWh"].idxmin()
    max_idx = df["Consumo_GWh"].idxmax()

    stats = [
        {"Indicador": "Observações mensais", "Valor": len(df), "Unidade": "meses"},
        {"Indicador": "Início da série", "Valor": df["Data"].min(), "Unidade": "data"},
        {"Indicador": "Fim da série", "Valor": df["Data"].max(), "Unidade": "data"},
        {"Indicador": "Consumo médio mensal", "Valor": df["Consumo_GWh"].mean(), "Unidade": "GWh"},
        {"Indicador": "Consumo mediano mensal", "Valor": df["Consumo_GWh"].median(), "Unidade": "GWh"},
        {"Indicador": "Desvio-padrão mensal", "Valor": df["Consumo_GWh"].std(ddof=1), "Unidade": "GWh"},
        {"Indicador": "Coeficiente de variação", "Valor": df["Consumo_GWh"].std(ddof=1) / df["Consumo_GWh"].mean() * 100, "Unidade": "%"},
        {"Indicador": "Mínimo mensal", "Valor": df.loc[min_idx, "Consumo_GWh"], "Unidade": "GWh"},
        {"Indicador": "Data do mínimo", "Valor": df.loc[min_idx, "Data"], "Unidade": "data"},
        {"Indicador": "Máximo mensal", "Valor": df.loc[max_idx, "Consumo_GWh"], "Unidade": "GWh"},
        {"Indicador": "Data do máximo", "Valor": df.loc[max_idx, "Data"], "Unidade": "data"},
        {"Indicador": "Períodos atípicos sinalizados", "Valor": len(outliers), "Unidade": "mês" if len(outliers) == 1 else "meses"},
    ]

    findings = [
        {
            "Aspecto": "Cobertura",
            "Resultado": f"Série completa com {len(df)} meses, de {df['Data'].min():%m/%Y} a {df['Data'].max():%m/%Y}",
            "Interpretação": "Não foram detectadas lacunas, duplicidades nem totais nacionais inválidos.",
        },
        {
            "Aspecto": "Evolução de longo prazo",
            "Resultado": f"O total anual variou {annual_change_pt}% entre {int(first_complete['Ano'])} e {int(last_complete['Ano'])}",
            "Interpretação": "A comparação utiliza somente anos com 12 meses completos; 2026 foi mantido como período parcial.",
        },
        {
            "Aspecto": "Sazonalidade média",
            "Resultado": f"Maior índice sazonal em {high_month['Mes']} e menor em {low_month['Mes']}",
            "Interpretação": "O índice foi calculado dentro de cada ano completo, reduzindo a influência do crescimento de longo prazo.",
        },
        {
            "Aspecto": "Períodos atípicos",
            "Resultado": f"Foi sinalizado {len(outliers)} mês pelo critério robusto aplicado aos resíduos da decomposição" if len(outliers) == 1 else f"Foram sinalizados {len(outliers)} meses pelo critério robusto aplicado aos resíduos da decomposição",
            "Interpretação": "Os meses foram apenas sinalizados; nenhum valor foi excluído e nenhuma causa foi atribuída sem evidência externa.",
        },
    ]

    monthly_columns = [
        "Data",
        "Ano",
        "Mes_Num",
        "Mes",
        "Consumo_MWh",
        "Consumo_GWh",
        "Consumidores",
        "Media_Movel_12m_GWh",
        "Total_Movel_12m_GWh",
        "Variacao_Mensal_pct",
        "Variacao_Anual_pct",
    ]
    decomposition_columns = [
        "Data",
        "Observado_GWh",
        "Tendencia_GWh",
        "Sazonal_GWh",
        "Residuo_GWh",
        "Z_Robusto",
        "Periodo_Atipico",
    ]
    outlier_columns = [
        "Data",
        "Consumo_GWh",
        "Tendencia_GWh",
        "Sazonal_GWh",
        "Residuo_GWh",
        "Z_Robusto",
        "Variacao_Anual_pct",
        "Direcao_Residuo",
    ]

    return {
        "metadata": {
            "source_file": input_path.name,
            "source_sheet": "Residencial Brasil",
            "source_url": "https://www.epe.gov.br/pt/publicacoes-dados-abertos/dados-abertos/dados-do-consumo-mensal-de-energia-eletrica",
            "analysis_start": df["Data"].min().strftime("%Y-%m-%d"),
            "analysis_end": df["Data"].max().strftime("%Y-%m-%d"),
            "observations": len(df),
            "missing_months": len(missing_dates),
            "duplicate_months": duplicate_dates,
            "invalid_national_totals": invalid_consumption,
            "outlier_method": outlier_method,
            "outlier_count": len(outliers),
            "decomposition_valid_start": decomposition.loc[decomposition["Tendencia_GWh"].notna(), "Data"].min().strftime("%Y-%m-%d"),
            "decomposition_valid_end": decomposition.loc[decomposition["Tendencia_GWh"].notna(), "Data"].max().strftime("%Y-%m-%d"),
        },
        "monthly": records(df.loc[:, monthly_columns]),
        "statistics": records(pd.DataFrame(stats)),
        "annual": records(annual),
        "seasonality": records(seasonality),
        "decomposition": records(decomposition.loc[:, decomposition_columns]),
        "outliers": records(outliers.loc[:, outlier_columns]),
        "findings": findings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_xlsx", type=Path)
    parser.add_argument("output_json", type=Path)
    args = parser.parse_args()

    result = analyse(args.input_xlsx)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["metadata"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
