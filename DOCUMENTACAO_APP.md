# Documentacao Oficial - AutoAnalista de Dados 2026

## 1) Visao geral

O **AutoAnalista de Dados 2026** e um app Streamlit para analise profissional de planilhas.
Ele foi projetado para receber arquivos de diferentes formatos, tratar os dados automaticamente,
avaliar qualidade, gerar insights gerenciais, produzir dashboards e exportar relatorios completos.

## 2) Objetivo do app

Transformar planilhas operacionais em inteligencia de negocio com:

- Tratativa automatica de dados
- Diagnostico de qualidade por pilares
- Analise exploratoria visual
- Insights executivos
- Relatorio tecnico e gerencial
- Opcao de Machine Learning

## 3) Tipos de arquivo suportados

- `.xlsx`
- `.xls`
- `.xlsm`
- `.xlsb`
- `.ods`
- `.csv`
- `.tsv`
- `.txt`

## 4) Fluxo funcional

1. Usuario faz login no app.
2. Usuario envia planilha e escolhe aba/tabela.
3. App executa conversao automatica de tipos (quando aplicavel).
4. Usuario define filtros, regras e tratativa de dados.
5. App calcula qualidade, KPIs, insights e dashboards.
6. Usuario pode executar ML opcional.
7. Usuario exporta evidencias em PDF/Excel/CSV.

## 5) Seguranca e governanca

- Perfis de acesso: `admin`, `analyst`, `viewer`
- Timeout de sessao por inatividade
- Bloqueio temporario apos tentativas invalidas de login
- Historico de execucoes com versionamento
- Regras customizadas persistidas por `usuario + area`

## 6) Tratativa automatica de dados

Antes da analise, o app pode:

- Remover linhas totalmente vazias
- Remover linhas duplicadas
- Remover outliers por IQR (com protecoes de seguranca)

Protecoes aplicadas:

- Limite maximo de remocao de outliers (%)
- Minimo de linhas apos tratativa
- Exclusao de colunas tipo ID da regra de outlier

## 7) Analise de qualidade de dados

O app calcula score global e semaforo de qualidade com 5 pilares:

- Completude
- Consistencia
- Unicidade
- Validade
- Atualidade

Entregas desta etapa:

- Score geral da base
- Score por pilar
- Semaforo por coluna
- Catalogo de problemas com prioridade

## 8) Dashboard gerencial

A aba Dashboard oferece:

- KPIs executivos
- Analise automatica textual da base
- Filtros internos do dashboard
- Graficos de pizza e barras por categoria
- Tendencia temporal (linha e barras)
- Distribuicao numerica (histograma/boxplot)
- Correlacao e dispersao entre metricas
- Lista dos principais riscos para gestao

## 9) Logica de identificadores (ID)

Colunas como `Chamado`, `Ticket`, `Protocolo` e similares sao tratadas como **ID**.
Por isso:

- Nao entram como metrica de soma
- Sao usadas como contagem/volume de registros unicos

Isso evita distorcoes em indicadores gerenciais.

## 10) Machine Learning (opcional)

### 10.1 ML sem alvo
- Clusterizacao automatica
- Deteccao de anomalias
- Projecao PCA para visualizacao

### 10.2 ML supervisionado
- Classificacao ou regressao (inferido pelo alvo)
- Validacao cruzada
- Deteccao de leakage
- Explicabilidade por permutation importance

## 11) Exportacoes

### PDF
- Resumo executivo
- Qualidade por pilares
- Insights e recomendacoes
- Analise do dashboard
- Resumo de ML

### Excel (pacote de evidencias)
- Dados brutos filtrados
- Dados tratados
- Resumo de qualidade
- Pilares e semaforo por coluna
- Catalogo de problemas
- Tratativa de dados
- Insights e recomendacoes
- Abas especificas do dashboard (`dash_*`)

### CSV
- Violacoes de regras

## 12) Estrutura tecnica principal

- `app.py`: logica de negocio e interface Streamlit
- `requirements.txt`: dependencias Python
- `app_data/`: historico e regras persistidas
- `README.md`: guia rapido do projeto

## 13) Como executar localmente

```powershell
cd C:\Users\walace.gorino\Downloads\autoanalista-excel-github
python -m venv .venv
.\.venv\Scripts\Activate
pip install -r requirements.txt
streamlit run app.py
```

## 14) Deploy recomendado gratuito

Para app Streamlit com backend Python persistente, o deploy recomendado e:

- **Streamlit Community Cloud** via GitHub

## 15) Assinatura do criador

**Criador do app: Walace.gorino**

---

Documento gerado para uso operacional, tecnico e gerencial do AutoAnalista de Dados 2026.
