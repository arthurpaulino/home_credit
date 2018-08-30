# home_credit

o projeto está ficando grande e para manter a organização achei melhor documentar
antes que vire um monstrão.

## descrição dos arquivos

### input/

esta pasta contém exatamente os arquivos `.csv` disponibilizados na competição.

### intermediary/

contém arquivos binários `.pkl` resultantes do processamento dos dados.

* principais arquivos: `train.pkl`, `test.pkl` e `corrs.pkl`

### output/

contém as saídas dos scripts. gráficos, pre-subs, blend, e metadados.

### data\_\*

scripts responsáveis pelo processamento dos dados.

`data_compile.py` depende do resultado da execução de `data_process.py`.

### features\_\*

responsáveis pelas heurísticas de seleção de features.

`features_score.py` cria o arquivo `output/features_metadata.csv`, que pode ser utilizado por heurísticas implementadas em `features_selection_functions.py`.

vale lembrar que para tal fim, também é interessante utilizar a matriz de correlações `intermediary/corrs.pkl`.

### model\_\*

`model_validate.py` faz a validação local de heurísticas de seleção de features
e de parâmetros de treinamento.

`model_gen_presubs.py` (precisa de mudanças profundas) gera os pre-subs a serem utilizados para o blend de
submissão.

`model_cv.py` executa uma validação rápida para 1 pre-sub.

### output\_\*

scripts que lidam com as saídas de `model_gen_presubs.py`.

`output_reset.py` limpa todos os arquivos e subpastas de `output/`.

### parameters.py

contém os parâmetros do ecossistema.
