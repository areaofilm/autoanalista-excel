# AutoAnalista de Dados 2026

App em Python (Streamlit) para analisar planilhas de varios formatos com foco em qualidade de dados.

## O que este app faz

- Upload de `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.ods`, `.csv`, `.tsv`, `.txt`
- Leitura de multiplas abas/tabelas
- Login com perfis e permissao (`admin`, `analista`, `viewer`)
- Seguranca de acesso com timeout de sessao e bloqueio temporario por tentativas invalidas
- Conversao automatica de tipos (numerico e data quando aplicavel)
- Filtros globais por coluna (numerico, data e categorico)
- Diagnostico profissional com 5 pilares de qualidade:
  - Completude
  - Consistencia
  - Unicidade
  - Validade
  - Atualidade
- Semaforo de qualidade por base e por coluna
- Regras customizaveis por coluna (required, email, cpf, non_negative, no_future_date, min/max)
- Regras persistidas por `usuario + area` e template automatico por dominio
- Catalogo de problemas com prioridade
- Insights executivos e recomendacoes de melhoria
- KPIs automaticos e comparativo de periodo atual vs anterior
- Graficos automaticos por tipo de dado (numerico, categorico e temporal)
- Correlacoes e heatmap
- Machine Learning sem alvo (clusterizacao + anomalias)
- Machine Learning supervisionado com validacao cruzada, deteccao de leakage e explicabilidade (permutation importance)
- Historico de analises e versionamento de relatorios
- Execucao sob demanda para performance (analise e ML so rodam quando acionados)
- Exportacao em PDF, Excel (pacote de evidencias) e CSV de violacoes

## Como executar

1. Abra o terminal na pasta do projeto:
   ```powershell
   cd C:\Users\walace.gorino\Downloads\autoanalista-excel
   ```

2. (Opcional) Crie e ative ambiente virtual:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate
   ```

3. Instale as dependencias:
   ```powershell
   pip install -r requirements.txt
   ```

4. Rode o app:
   ```powershell
   streamlit run app.py
   ```

## Fluxo de uso

1. Faca login.
2. Faca upload da planilha.
3. Escolha a aba/tabela desejada (quando houver mais de uma).
4. Aplique filtros globais e regras por coluna (opcional).
5. Navegue pelas abas: `Resumo`, `Qualidade`, `Insights`, `ML`, `Relatorio`.
6. Exporte evidencias em PDF/Excel/CSV.

## Usuarios padrao

- `admin` / `admin2026!`
- `analista` / `analista2026!`
- `viewer` / `viewer2026!`

## Observacoes

- Para analises de ML mais confiaveis, use planilhas com mais de 30 linhas.
- Colunas numericas e de data melhoram a qualidade dos insights automaticos.
