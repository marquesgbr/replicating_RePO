# Novo dataset (`data/new_data/`)

Este documento descreve o dataset adicional usado para avaliar a metodologia
RePO/RePO+ fora dos dados do artigo de referência. A fonte canônica é o ZIP
`data/new_data/New-dataset-split-clean.zip`, versionado via Git LFS. O script
`data/new_data/preprocess_new_data.py` é legado: ele não reproduz o ZIP e não
é necessário para executar a avaliação.

> **Escopo:** a implementação e os comandos vivem em `code/new_data_eval/`.
> Nada deste fluxo altera os resultados de fluxo/pacotes do artigo nem usa
> `cluster/`.

## Fonte e citação

**PENDENTE DE CONFIRMAÇÃO OFICIAL.** O nome do ZIP, o script legado e o
cabeçalho dos CSVs indicam o **CIC-ToN-IoT**. A origem oficial, a licença, a
citação bibliográfica e a justificativa formal da escolha ainda precisam ser
confirmadas com quem produziu o branch. Nenhuma referência é inventada aqui;
esta seção deve ser atualizada antes da redação final do relatório.

- Arquivo canônico: `data/new_data/New-dataset-split-clean.zip` (Git LFS).
- Conteúdo declarado: os CSVs já processados e divididos, **não** o dataset
  bruto.
- O ZIP é a fonte de execução; não se recria o ZIP a partir do dataset bruto
  nesta entrega.

## Conteúdo do ZIP

| Item | Descrição |
| --- | --- |
| `new_train.csv` | Parte benigna reservada ao treino/reprodução do autoencoder; ~896 MB descomprimido. |
| `new_test.csv` | 30% dos benignos mais todas as amostras maliciosas; ~2,1 GB descomprimido. |

Os dois CSVs têm o mesmo layout de **85 colunas**, na ordem canônica:

| Índices | Conteúdo |
| --- | --- |
| `0` | `column00`, contador de linha criado no pré-processamento |
| `1`–`7` | identificadores e metadados (`Flow ID`, IPs, portas, protocolo, timestamp) |
| `8`–`83` | 76 features numéricas de fluxo |
| `84` | `Label` |

O cabeçalho é validado antes de qualquer leitura de dados. O CSV de treino deve
conter apenas `BENIGN`; o de teste deve conter `BENIGN` e as nove classes
maliciosas.

### Classes

Rótulos canônicos: `BENIGN`, `xss`, `password`, `injection`, `scanning`,
`backdoor`, `ransomware`, `mitm`, `ddos` e `dos`. A comparação ignora caixa e
espaços nas extremidades; qualquer rótulo desconhecido aborta a validação.

As contagens reais por classe são **impressas pelo comando `validate`** e
registradas nos JSONs de resultado da avaliação normal. A tabela abaixo é
apenas o esqueleto que a Task 9 preenche com os números do dataset extraído;
não há contagens hardcoded neste documento.

| Classe | Contagem |
| --- | --- |
| `BENIGN` | impressa por `validate` |
| `xss` | impressa por `validate` |
| `password` | impressa por `validate` |
| `injection` | impressa por `validate` |
| `scanning` | impressa por `validate` |
| `backdoor` | impressa por `validate` |
| `ransomware` | impressa por `validate` |
| `mitm` | impressa por `validate` |
| `ddos` | impressa por `validate` |
| `dos` | impressa por `validate` |

Classes com menos de 100 amostras aparecem nos resultados com a marca
`low_n: true` e também numa média macro de sensibilidade, para evitar
conclusões fortes sobre amostras pequenas.

## Split

O split do ZIP segue a regra do pré-processamento:

- os benignos são divididos em **70% treino / 30% teste**;
- `new_train.csv` recebe somente os 70% benignos;
- `new_test.csv` recebe os 30% benignos restantes mais **todas** as amostras
  maliciosas;
- as linhas do teste são embaralhadas com seed fixa.

Assim, o threshold é calibrado em benignos que não foram vistos no treino. O
split separa apenas benignos: não há validação separada nem divisão por classe
maliciosa. O script de treino rejeita qualquer rótulo diferente de `BENIGN` no
CSV de treino.

## Features

O autoencoder consome **77 features**, nesta ordem:

1. `Dst Port` (índice 5 do CSV);
2. as 76 colunas de fluxo dos índices 8–83.

A lista canônica dos nomes está em `FEATURE_NAMES`
(`code/new_data_eval/data.py`). Os dados são carregados **crus**; a
normalização acontece apenas na pontuação:

```text
x_norm = (x - train_min) / (train_max - train_min + 1e-6)
```

`train_min` e `train_max` são os arrays commitados em
`data/new_data/x_train_meta/` (77 valores cada). Linhas com `inf`/`nan` nas 77
features são descartadas e contadas; a validação falha se o modelo não aceitar
77 entradas.

### Features constantes no treino

Oito features têm `train_max == train_min` no split de treino. Como o
denominador da normalização vira apenas o epsilon, essas features produzem
valores muito grandes; os índices são reportados por `validate` para que a
limitação fique visível:

| Índice no modelo | Nome |
| --- | --- |
| `31` | `Bwd PSH Flags` |
| `32` | `Fwd URG Flags` |
| `33` | `Bwd URG Flags` |
| `48` | `URG Flag Cnt` |
| `55` | `Fwd Byts/b Avg` |
| `56` | `Fwd Pkts/b Avg` |
| `57` | `Fwd Blk Rate Avg` |
| `63` | `Subflow Bwd Pkts` |

## Protocolo

A avaliação reutiliza o protocolo RePO/RePO+ do artigo:

- **RePO:** média dos erros de reconstrução das máscaras;
- **RePO+:** os erros são divididos em cinco grupos iguais, toma-se o mínimo
  de cada grupo e somam-se os cinco mínimos (100 máscaras → 5 grupos de 20);
- cada feature é mascarada com probabilidade **0,75**;
- as máscaras são derivadas de forma determinística de
  `(seed, sample_index, phase, repeat)`, com seed padrão **2019**, para que
  `--batch-size` e o tamanho do chunk não alterem os escores.

### Avaliação normal

`normal` pontua todas as linhas válidas de `new_test.csv` com RePO e RePO+,
calcula os thresholds nos escores benignos do **FPR 1%** e **10%** e grava um
JSON por modo:

```text
results/new_data/normal/repo.json
results/new_data/normal/repo_plus.json
```

Cada JSON contém thresholds, FPR medido, TPR por classe, contagens, médias
macro/weighted apenas nas nove classes maliciosas, tempo de execução, hashes de
dados/modelo, commit e versões do ambiente.

### Avaliação adversarial

`adversarial` executa uma classe por vez ou `--attack all`. Por padrão:

- usa até **5.000 amostras aleatórias** por ataque, com seed fixa (se houver
  menos, usa todas);
- lê o threshold de FPR 10% do `normal/repo_plus.json` (se o JSON não existir,
  recalcula e registra aviso);
- otimiza a loss RePO (soma dos 100 erros médios), mas mede sucesso e TPR da
  vítima com o agregado RePO+;
- grava checkpoint a cada 50 amostras e retoma sem duplicar amostras.

Resultados principais e sensibilidades ficam separados:

```text
results/new_data/adversarial/primary/<ataque>.json
results/new_data/adversarial/sensitivity_first_n/<ataque>.json
results/new_data/adversarial/sensitivity_paper_0.8/<ataque>.json
```

O resultado principal é sempre `variant="primary"`. Arquivos `.npz` de scores
e checkpoints são artefatos locais e estão ignorados pelo Git.

## Comandos

Execute a partir da raiz do repositório, com o ambiente virtual ativo. O pacote
é importado como `new_data_eval`; por isso `PYTHONPATH=code` é obrigatório e
**não** existe `code/__init__.py`.

```bash
# 1. Baixar o ZIP LFS, extrair e conferir os CSVs
bash extract_new_data.sh

# 2. Validar cabeçalho, rótulos, metadados e modelo
PYTHONPATH=code python -m new_data_eval validate

# 3. Avaliação normal completa (RePO e RePO+)
PYTHONPATH=code python -m new_data_eval normal --modes repo,repo_plus

# 4. Um ataque, depois todos
PYTHONPATH=code python -m new_data_eval adversarial --attack ddos --variant primary
PYTHONPATH=code python -m new_data_eval adversarial --attack all --variant primary

# 5. Smoke rápido (validação + normal + um ataque em results/new_data/smoke/)
PYTHONPATH=code python -m new_data_eval smoke

# 6. Treino reproduzível do autoencoder
python code/train_new_data_model.py \
  --train-csv data/new_data/new_train.csv \
  --output-dir models/new_data_model \
  --seed 2019 --epochs 5 --batch-size 256
```

Para inspeção rápida sem percorrer o dataset inteiro, use `--max-samples` e
`--n-masks` (o número de máscaras deve ser múltiplo de 5 para RePO+):

```bash
PYTHONPATH=code python -m new_data_eval normal \
  --modes repo,repo_plus --max-samples 20 --n-masks 5 --batch-size 2
PYTHONPATH=code python -m new_data_eval adversarial \
  --attack ddos --max-samples 5 --n-masks 5 --variant primary
```

Os notebooks `code/new_data_RePO_Normal_Test.ipynb` e
`code/new_data_RePO_Adversarial.ipynb` são wrappers finos dessa CLI: eles não
reimplementam escores, thresholds nem o ataque e aceitam os mesmos parâmetros
de smoke.

O treino grava `training_manifest.json` no diretório de saída com o hash
SHA-256 do CSV, seed, versões de Python/NumPy/pandas/TensorFlow/Keras, número
de epochs, batch size, loss final e shape da matriz de treino.

## Limitações

- **Fonte e citação pendentes:** a atribuição oficial ao CIC-ToN-IoT depende de
  confirmação externa (ver “Fonte e citação”).
- **Calibração e medição no mesmo conjunto:** os thresholds normal e
  adversarial são calculados nos benignos de `new_test.csv` e o FPR é medido
  nesses mesmos benignos, como no protocolo do artigo. Isso é uma limitação
  registrada, não um erro acidental.
- **Proveniência do modelo:** os hashes e o manifesto provam o que foi
  executado, mas não provam que o modelo commitado veio do treino original. A
  validação de sanidade (erro benigno menor que ruído) não substitui essa
  prova; o retreino é uma etapa explícita e separada.
- **Features constantes:** as oito features listadas acima são normalizadas
  apenas pelo epsilon e podem dominar escores; os índices são reportados em
  toda validação.
- **Classes pequenas:** classes com `n < 100` recebem `low_n` e não devem
  sustentar conclusões fortes.
- **Custo computacional:** a avaliação completa percorre ~2,1 GB de teste e
  repete 100 máscaras por amostra; use GPU quando disponível e preserve os
  checkpoints para retomar interrupções.
- **Sem reconstrução do dataset bruto:** o ZIP processado é a fonte; o script
  `preprocess_new_data.py` é legado e não deve ser tratado como reprodutor do
  ZIP.
