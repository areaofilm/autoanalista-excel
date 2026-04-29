# AutoAnalista de Dados 2026

Aplicacao em Python (Streamlit) para analise profissional de planilhas, com tratativa de dados, dashboard gerencial, regras de qualidade, insights, ML e relatorios.

## Documentacao

- Documentacao completa: `DOCUMENTACAO_APP.md`
- Arquivo principal do app: `app.py`
- Dependencias: `requirements.txt`

## Funcionalidades principais

- Upload de planilhas (`xlsx`, `xls`, `xlsm`, `xlsb`, `ods`, `csv`, `tsv`, `txt`)
- Tratativa automatica (linhas vazias, duplicadas e outliers)
- Dashboard analitico com filtros e visualizacoes (pizza, barras, tendencia, correlacao)
- Diagnostico de qualidade em 5 pilares
- Regras por coluna com persistencia por usuario e area
- Insights executivos, catalogo de problemas e plano de acao
- ML opcional (sem alvo e supervisionado)
- Exportacao PDF + Excel de evidencias + CSV

## Execucao local

```powershell
cd C:\Users\walace.gorino\Downloads\autoanalista-excel-github
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py
```

## Assinatura do criador

Criador do app: **Walace.gorino**
