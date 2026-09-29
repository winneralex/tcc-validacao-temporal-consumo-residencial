# Validação temporal de modelos para previsão do consumo residencial de eletricidade no Brasil

Repositório dos códigos utilizados no Trabalho de Conclusão de Curso de Luis Alexander Pérez, desenvolvido no MBA em Data Science e Analytics da USP/Esalq, sob orientação de Felipe Pinto Da Silva.

O estudo compara o método ingênuo sazonal, Holt-Winters e SARIMA utilizando uma série mensal da Empresa de Pesquisa Energética (EPE), de janeiro de 2004 a junho de 2026.

## Estrutura

```text
dados/                  Base filtrada da EPE
src/                    Scripts da análise
resultados_referencia/  Resultados utilizados no TCC
```

## Ordem de execução

```bash
python src/01_analise_exploratoria.py dados/Base_Oficial_Filtrada_Consumo_Residencial_Brasil_2004_2026-06.xlsx analise_exploratoria.json
python src/02_comparacao_inicial.py dados/Base_Oficial_Filtrada_Consumo_Residencial_Brasil_2004_2026-06.xlsx
python src/03_validacao_origem_movel.py dados/Base_Oficial_Filtrada_Consumo_Residencial_Brasil_2004_2026-06.xlsx
python src/04_previsao_final_bootstrap.py dados/Base_Oficial_Filtrada_Consumo_Residencial_Brasil_2004_2026-06.xlsx
```

O terceiro script realiza 405 ajustes SARIMA e pode levar alguns minutos.

## Ambiente de referência

- Python 3.13.14
- pandas 2.3.3
- NumPy 2.3.5
- SciPy 1.16.3
- statsmodels 0.14.6

## Resultados principais

- Holt-Winters selecionado na validação por origem móvel;
- MAPE agregado de 3,28%;
- previsão de 189,65 TWh entre julho de 2026 e junho de 2027;
- crescimento previsto de 3,76%;
- intervalos empíricos de 95% por bootstrap em blocos.

Fonte dos dados: https://www.epe.gov.br/pt/publicacoes-dados-abertos/dados-abertos/dados-do-consumo-mensal-de-energia-eletrica
