# Consolidação da replicação do RePO no cluster Apuana

> Documento gerado durante a execução no cluster Apuana (CIn/UFPE).
> Resultados normais completos; resultados adversariais de fluxo e pacotes
> conforme o andamento indicado em cada seção.

## 1. Resumo executivo

- O pipeline do RePO foi reproduzido no cluster Apuana com TensorFlow 2.21,
  Keras 3.12.4 e uma GPU **A100 80 GB**.
- A avaliação **normal** (FPR alvo de 0,01) reproduziu o RePO+ com alta
  fidelidade: **0,2525** no fluxo (artigo: 0,2549) e **0,3472** nos pacotes
  (artigo: 0,3477).
- O RePO (sem agregação por grupos) ficou **acima** do artigo nos dois modelos,
  o que reduz o ganho atribuído ao RePO+ nesta replicação.
- Os thresholds adversariais publicados (0,016235… no fluxo e 0,081066… nos
  pacotes) reproduziram **FPR ≈ 0,10** nos nossos dados (0,1005 e 0,0997),
  validando a calibração do artigo.
- Foram encontrados problemas de metodologia e de código que afetam a
  interpretação dos números: falta de normalização no notebook normal de
  fluxo, métrica de pacotes duplicada, comparação de rótulos com dtype
  incompatível, descarte do último lote e, principalmente, **viés de amostragem
  por usar as primeiras N amostras** de cada ataque.

## 2. Ambiente e método

| Item | Valor |
|---|---|
| Cluster | Apuana (CIn/UFPE), nó `cluster-node3` |
| GPU | NVIDIA A100 80 GB PCIe, driver 590.48.01 |
| Python | 3.10.12 (`/usr/bin/python3.10`) |
| TensorFlow | 2.21.0 (`tensorflow[and-cuda]`) |
| Keras | 3.12.4 |
| NumPy / Pandas | 2.2.6 / 2.2.3 |
| Seed | 2019 |
| Dados | CICIDS2017 (fluxo e pacotes) |
| Fluxo: treino / teste | Monday / Tuesday–Friday |
| Pacotes: treino / teste | Monday / Tuesday–Friday |

Procedimento preservado dos notebooks:

- 100 máscaras por amostra, cada feature mascarada com probabilidade 0,75;
- **RePO** = média do erro de reconstrução das 100 máscaras;
- **RePO+** = mínimo por grupo de 20 máscaras, somando os 5 mínimos;
- fluxo: 2.298.431 amostras de teste;
- pacotes: 2.216.623 janelas válidas de 20 pacotes.

## 3. Avaliação normal (FPR alvo = 0,01)

### 3.1 Resumo

| Modelo | Modo | Threshold | FPR obtido | TPR médio | Artigo | Δ |
|---|---|---|---|---|---|---|
| Fluxo | RePO | 0,041301 | 0,0100 | 0,2443 | 0,2161 | +0,0282 |
| Fluxo | RePO+ | 0,106084 | 0,0100 | **0,2525** | **0,2549** | −0,0024 |
| Pacotes | RePO | 0,035509 | 0,0100 | 0,3185 | 0,2765 | +0,0420 |
| Pacotes | RePO+ | 0,142918 | 0,0100 | **0,3472** | **0,3477** | −0,0005 |

O FPR no threshold adversarial publicado também foi medido:

| Modelo | Threshold do artigo | FPR resultante |
|---|---|---|
| Fluxo (RePO) | 0,016235964372754097 | 0,1005 |
| Fluxo (RePO+) | 0,016235964372754097 | 0,0997 |
| Pacotes (RePO+) | 0,08106673508882523 | 0,0997 |

Esse é um teste forte de fidelidade: os thresholds absolutos do artigo
produzem exatamente o FPR = 0,10 declarado.

### 3.2 TPR por ataque — fluxo

| Ataque | n | RePO | RePO+ |
|---|---|---|---|
| FTP-Patator | 7.935 | 0,0000 | 0,0000 |
| SSH-Patator | 5.897 | 0,0000 | 0,0003 |
| DoS slowloris | 5.796 | 0,1917 | 0,1938 |
| DoS Slowhttptest | 5.499 | 0,0100 | 0,1020 |
| DoS Hulk | 230.124 | 0,6164 | 0,6239 |
| DoS GoldenEye | 10.293 | 0,1797 | 0,2230 |
| Heartbleed | 11 | 1,0000 | 1,0000 |
| Infiltration | 36 | 0,3611 | 0,2222 |
| Bot | 1.956 | 0,0102 | 0,0148 |
| PortScan | 158.839 | 0,0001 | 0,0004 |
| DDoS | 128.025 | 0,3177 | 0,3974 |

### 3.3 TPR por ataque — pacotes

| Ataque | n | RePO | RePO+ |
|---|---|---|---|
| FTP-Patator | 5.537 | 0,1651 | 0,1875 |
| SSH-Patator | 6.931 | 0,0004 | 0,0006 |
| Slowloris | 2.380 | 0,4689 | 0,4887 |
| Slowhttptest | 1.963 | 0,5777 | 0,7275 |
| Hulk | 112.276 | 0,1935 | 0,2649 |
| GoldenEye | 5.309 | 0,0812 | 0,1053 |
| Heartbleed | 2.465 | 0,0000 | 0,0004 |
| Infiltration | 10.496 | 0,7969 | 0,7992 |
| Botnet | 494 | 0,0000 | 0,0000 |
| PortScan | 16.203 | 0,9931 | 0,9931 |
| DDoS | 64.030 | 0,2273 | 0,2525 |

## 4. Problemas de metodologia e de código evidenciados

### 4.1 Viés de amostragem: "primeiras N amostras"

Os notebooks e o artigo usam as **primeiras 5.000 amostras** de cada ataque de
fluxo e os **primeiros 25.000 pacotes** de cada ataque de pacotes. Isso não é
uma amostra aleatória: o tráfego do CICIDS2017 é capturado em ordem temporal e
as classes se concentram em janelas de tempo.

Medimos o efeito comparando o TPR no conjunto completo com o TPR nas primeiras
N amostras e com 20 sorteios aleatórios de N amostras:

| Ataque (fluxo) | n | TPR completo (RePO+) | TPR primeiras 5.000 | TPR 5.000 aleatórias |
|---|---|---|---|---|
| DoS Hulk | 230.124 | 0,6239 | 0,6220 | 0,6238 ± 0,0043 |
| PortScan | 158.839 | 0,0004 | 0,0004 | 0,0004 ± 0,0002 |
| **DDoS** | 128.025 | **0,3974** | **0,2890** | **0,3965 ± 0,0070** |
| DoS GoldenEye | 10.293 | 0,2230 | 0,2376 | 0,2223 ± 0,0042 |
| DoS slowloris | 5.796 | 0,1938 | 0,2246 | 0,1942 ± 0,0021 |
| DoS Slowhttptest | 5.499 | 0,1020 | 0,1122 | 0,1023 ± 0,0013 |

No RePO, o DDoS cai de 0,3177 (completo) para 0,1898 (primeiras 5.000), uma
queda relativa de **40%**. Ou seja, a escolha das primeiras amostras pode
subestimar ou superestimar ataques inteiros, e o artigo não reporta intervalo
de confiança nem descreve esse critério.

Para pacotes, a maioria dos ataques tem menos de 25.000 amostras, então o
recorte não muda nada. As exceções:

| Ataque (pacotes) | n | TPR completo (RePO+) | TPR primeiros 25.000 |
|---|---|---|---|
| Hulk | 112.276 | 0,2649 | 0,2702 |
| DDoS | 64.030 | 0,2525 | 0,2522 |

O viés é pequeno aqui, mas o mesmo problema conceitual existe.

### 4.2 Notebook normal de fluxo não normaliza os dados

No `Flow_based_RePO_Normal_Test.ipynb` a normalização min–max está comentada,
embora o treino normalize. O runner do cluster normaliza sempre. Evidência
indireta de que a normalização é o procedimento correto: com ela, o RePO+ fica
em 0,2525 contra 0,2549 do artigo. Sem ela, os escores não são comparáveis aos
do treino e o threshold publicado perde sentido.

Observação: o threshold antigo do notebook (0,106045) é quase idêntico ao nosso
threshold normalizado (0,106084), o que explica por que o erro passou
despercebido — a escala global muda pouco, mas a ordenação por amostra muda.

### 4.3 Métrica adversarial de pacotes duplicava o TPR normal

No `Packet_based_RePO_Adversarial.ipynb`, o resultado gravava
`attacker_tpr = normal_tpr`, isto é, o TPR **antes** do ataque era reportado
como se fosse o TPR **depois**. Isso torna impossível medir a queda de detecção
a partir do JSON do notebook. O runner passou a gravar campos distintos:
`normal_tpr`, `victim_tpr`, `attack_success_rate`, `fooled_rate` e `counts`.

### 4.4 Rótulos de pacotes são `float64`

Os arquivos `labels.npy` do dataset de pacotes vêm como `float64`, não inteiros.
Uma comparação `labels == índice` funciona, mas `labels == "Benign"` falha
silenciosamente e produz `count=0` para todas as classes. Foi exatamente o que
aconteceu na primeira execução: o threshold foi calculado corretamente, mas
todas as taxas por classe saíram nulas. Corrigimos aceitando qualquer dtype
numérico; o problema indica fragilidade do código original, que assume um dtype
que o dataset não garante.

### 4.5 Descarte do último lote parcial

O notebook normal de pacotes descarta o último lote parcial de 50 janelas. O
runner pontua todas. O impacto é pequeno (no máximo 49 janelas em ~2,2
milhões), mas é uma fonte silenciosa de diferença entre execuções.

### 4.6 Broadcasting implícito nas máscaras de pacotes

O notebook de pacotes depende de broadcasting entre o lote e as 100 máscaras,
o que produz um número de máscaras efetivas diferente do declarado. O runner
aplica explicitamente 100 máscaras por janela, como descrito no artigo. Essa é
uma diferença de implementação que pode explicar parte da diferença de TPR
entre o notebook e o artigo.

### 4.7 Threshold adversarial do fluxo tem fator 0,8 arbitrário

O notebook usa `thr = real_thr * 0.8` para compensar o efeito das features do
grupo 4. O fator não é derivado de um princípio: é um ajuste empírico. Nossos
resultados mostram que ele **funciona** (FPR resultante 0,1005), mas o valor
depende do modelo, do conjunto e da escala dos escores; um modelo retreinado
exigiria recalibrar o fator.

### 4.8 Classes com pouquíssimas amostras

No fluxo, Heartbleed tem **11** amostras e Infiltration **36**. Um TPR de 1,0
em Heartbleed significa 11 acertos; o intervalo de confiança é enorme. O artigo
reporta TPR por ataque sem indicar o tamanho amostral, o que dá uma falsa
impressão de precisão. Nos pacotes, Botnet tem 494 amostras e Heartbleed 2.465.

### 4.9 Índices de features codificados manualmente

O ataque de fluxo usa índices fixos (por exemplo, 61→2, 63→3, 62→4, 64→5) e
listas de features "inteiras", "flags" e "agregadas". Isso depende da ordem
exata das 77 colunas do CICIDS2017 e quebra se o CSV for reprocessado com outra
ordem ou outro conjunto de features. O mesmo vale para os limites `packet_mins`
e `packet_maxs` do ataque de pacotes.

### 4.10 Ausência de seed nos notebooks

Os notebooks não fixam seed para as máscaras aleatórias. Duas execuções do
mesmo notebook podem produzir TPRs diferentes, especialmente nos ataques com
poucas amostras. O runner usa `SEED=2019` e registra o seed no manifesto.

## 5. Comparação com o artigo

| Afirmação do artigo | O que observamos |
|---|---|
| RePO+ atinge 25,49% no fluxo (FPR 0,01) | 25,25% — confere |
| RePO+ atinge 34,77% nos pacotes (FPR 0,01) | 34,72% — confere |
| RePO atinge 21,61% no fluxo | 24,43% — acima |
| RePO atinge 27,65% nos pacotes | 31,85% — acima |
| Thresholds de FPR 0,10 calibrados | FPR 0,1005 e 0,0997 — confere |
| RePO+ melhora sobre RePO | melhora, mas menor que no artigo |
| Inferência de 17.550 fluxos/s (Titan V) | ~33.000 fluxos/s (A100) |
| Inferência de 3.285 pacotes/s (Titan V) | ~12.000 pacotes/s (A100) |

O ganho do RePO+ sobre o RePO nesta replicação foi de **+0,82 ponto
percentual** no fluxo e **+2,87 pontos** nos pacotes, contra +3,88 e +7,12
pontos no artigo. A diferença não invalida o método, mas mostra que o tamanho
do ganho é sensível ao treino, às máscaras e ao conjunto de avaliação.

## 6. Resultados adversariais

### 6.1 Fluxo

> Preenchido ao final do array `18701` (11 ataques, 5.000 amostras cada).

### 6.2 Pacotes

> Preenchido após o array de pacotes (11 ataques, 25.000 pacotes cada).

## 7. Reprodutibilidade

- Branch: `feat/apuana-cluster`.
- Comandos e decisões: `cluster/README.md` e `docs/APUANA_RUNBOOK.md`.
- Ambiente exato: `$HOME/replicating-repo-results/env-lock.txt`.
- Cada resultado JSON contém `config_hash`, impressão digital do modelo,
  hash dos arquivos de normalização, commit do Git e manifesto do ambiente.
- Checkpoints a cada 50 amostras/pacotes permitem retomar sem perder trabalho.

## 8. Limitações desta replicação

1. Os resultados usam os modelos pré-treinados dos autores; não retreinamos.
2. A avaliação adversarial usa as primeiras N amostras, como o artigo, e
   portanto herda o viés da seção 4.1.
3. O ataque adversarial de fluxo reavalia cada amostra com uma única máscara
   aleatória, o que adiciona variância ao TPR de vítima.
4. Não comparamos com as baselines do artigo (Kitsune, DAGMM, BiGAN).
5. O cluster tinha nós com defeito e limites de 2 GPUs por usuário, o que
   serializou parte dos experimentos.

## 9. Conclusões

1. **O RePO+ é reproduzível**: os dois números centrais do artigo bateram com
   erro abaixo de 0,3 ponto percentual.
2. **O RePO puro ficou acima do artigo**, reduzindo o ganho do RePO+. Isso
   merece discussão no relatório: o benefício do RePO+ não é uniforme entre
   modelos e conjunto de teste.
3. **Os thresholds publicados são válidos**: reproduziram FPR = 0,10, o que
   sustenta a comparação adversarial.
4. **O procedimento de amostragem "primeiras N" é uma fragilidade
   metodológica** com efeito grande em pelo menos um ataque (DDoS no fluxo).
5. **Os notebooks têm erros que não aparecem nos números do artigo** (métrica
   duplicada, normalização comentada, dtype de rótulos), o que recomenda usar
   o runner do cluster como implementação de referência para novos
   experimentos.
