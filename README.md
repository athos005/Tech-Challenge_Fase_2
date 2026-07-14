# Classificação da Qualidade de Vinhos com Machine Learning
**Tech Challenge - Fase 2 | POSTECH - Data Analytics & Artificial Intelligence**

## Sobre o Projeto
A avaliação da qualidade de um vinho é tradicionalmente feita por especialistas através de análises sensoriais. Embora eficaz, este processo é moroso, dispendioso e sujeito à subjetividade. 

Este projeto visa resolver esse problema de negócio através do desenvolvimento de um modelo preditivo de **Machine Learning**. Utilizando dados físico-químicos recolhidos durante a produção (como acidez, teor alcoólico, densidade e níveis de dióxido de enxofre), o modelo classifica os vinhos em duas categorias:
* **Alta Qualidade (1):** Vinhos com nota igual ou superior a 7.
* **Baixa/Média Qualidade (0):** Vinhos com nota inferior a 7.

## Estrutura do Repositório
O projeto está organizado de forma modular, separando as fases de análise, processamento e modelagem:

wine-quality-classification/
- data/                  _# Base de dados original (WineQT.csv) e dados processados_
- notebooks/             _# Notebook interativo com Análise Exploratória (EDA) e Storytelling_
  - analise_e_modelagem.ipynb
- src/                   _# Scripts auxiliares Python para automação do pipeline_
  - wine_processing.py
  - wine_modeling.py
- results/               _# Gráficos, matrizes de confusão e métricas exportadas_
  - requirements.txt       _# Bibliotecas e dependências do projeto_
- README.md              _# Documentação principal_

---

## Como Executar o Projeto

Para garantir que o código corre perfeitamente no seu computador ou no ambiente de avaliação, siga o passo a passo abaixo:

### Pré-requisitos
* Ter o Python (versão 3.8 ou superior) instalado.
* (Opcional, mas recomendado) Ter o Git instalado para clonar o repositório.

### Passo 1: Obter o código
Clone este repositório para a sua máquina local ou faça o download do ficheiro ZIP e extraia os ficheiros.

### Passo 2: Criar um Ambiente Virtual (Recomendado)
Para evitar conflitos de versão com outras bibliotecas da sua máquina, crie e ative um ambiente virtual:
* No Windows:
  python -m venv venv
  venv\Scripts\activate
* No Linux/Mac:
  python3 -m venv venv
  source venv/bin/activate

### Passo 3: Instalar as Dependências
Com o ambiente virtual ativado, instale todas as bibliotecas necessárias executando o comando abaixo na raiz do projeto:
pip install -r requirements.txt

### Passo 4: Executar a Análise (Jupyter Notebook)
Se deseja visualizar o storytelling dos dados, a Análise Exploratória (EDA) e os gráficos de forma interativa:
1. Abra o terminal e digite `jupyter notebook` (ou abra a pasta do projeto no VS Code).
2. Navegue até à pasta `notebooks/`.
3. Abra o ficheiro `wine_analysis_modeling.ipynb` e execute as células sequencialmente.

### Passo 5: Executar o Pipeline de Produção (Scripts)
Se deseja correr a versão de automação (ideal para testes rápidos e validação do modelo), execute os scripts através do terminal a partir da raiz do projeto:

1. Preparar e limpar os dados:
python src/wine_processing.py
(Isto vai gerar os ficheiros de treino e teste dentro da pasta data/)

2. Treinar os modelos e gerar resultados:
python src/wine_modeling.py
(Após a execução, todas as métricas, relatórios de precisão e gráficos serão guardados automaticamente dentro da pasta results/)
---

## 📊 Principais Insights e Resultados
* Desbalanceamento Resolvido: Apenas ~14% da base original correspondia a vinhos premium. A utilização do algoritmo Random Forest com a técnica de balanceamento de classes (class_weight='balanced') revelou-se a melhor estratégia para capturar a classe minoritária (alta qualidade) sem perder robustez geral.
* Fatores Críticos de Qualidade: A análise de Feature Importance destacou que o Teor Alcoólico, a Acidez Volátil (que deve ser rigorosamente controlada para evitar o "sabor a vinagre") e os Sulfatos são as características químicas mais determinantes para que um vinho receba notas máximas dos enólogos.

---
Autores:
* Athos Phelipe dos Santos
