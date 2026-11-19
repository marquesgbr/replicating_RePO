# Cascata *cheap-first* e decisão sequencial

> **Escopo.** Este documento descreve o método, o algoritmo e os resultados da **direção B** do
> estudo de eficiência de máscaras do RePO: alocar máscaras **por amostra** em vez de fixar um
> número para todas.  A referência de comparação é o detector RePO+ do artigo com **100 máscaras**
> (além do baseline de média simples de 100 máscaras usado pela regra sequencial), e a referência interna dos experimentos é o `full_repo_plus` do próprio
> `cascade_flow.json` (mesmo conjunto de erros, 5 repetições).
>
> **Fonte dos números.** `$HOME/replicating-repo-results/mask_efficiency/` no cluster
> `glp3@slurm-client1.cin.ufpe.br`, copiado para `/tmp/opencode/mask_full/`
> (`baseline_flow.json`, `cascade_flow.json`, `cascade_attack/*.json`, com `variance_flow.json`
> como contexto da direção A).
> Nenhum número foi estimado: todos saem dos JSONs ou de aritmética explícita sobre eles.

---

## 1. Resumo executivo

- A cascata **cheap-first** com estágio barato de **10 máscaras** e orçamento de **1 % de falsos
  negativos** reduz o custo de **100 → 60,8 máscaras por amostra (−39,2 %)** na mistura de
  avaliação (99 % benigno / 1 % ataque), mantendo **exatamente** a TPR do RePO+ completo na mesma
  rodada (macro 0,2560; ponderada 0,1871) e o mesmo ponto de operação (FPR de 1 %).
- **43,98 %** dos benignos são despachados só com o estágio barato; os outros 56,02 % pagam as
  100 máscaras.  Os ataques quase nunca pulam: o estágio barato descarta 1,0004 % deles.
- No protocolo adversarial (11 ataques, 9.047 amostras no total), o **bypass agregado** do
  estágio barato é **0,25 % (23/9.047)** com 10 máscaras e **0,02 % (2/9.047)** com 25.  Excluindo
  o SSH-Patator — que já é falso negativo em massa no ponto de operação — o bypass cai para
  **0,04 % (3/8.047)** com 10 máscaras.
- A decisão sequencial é muito mais agressiva: **5,32 máscaras/amostra (−94,7 %)**, com 99,93 %
  de concordância com o RePO de 100 máscaras, mas opera sobre o RePO (não sobre o RePO+) e custa
  **−1,2 ponto percentual** de TPR macro.
- **A cascata não melhora a TPR: ela preserva a TPR gastando menos.**  O ganho é de custo e é
  **dependente da carga**: com 100 % de benignos o custo cai para 60,4; com 100 % de ataques ele
  sobe para 99,1.

---

## 2. Problema e motivação

### 2.1 Por que 100 máscaras custam caro

O detector de fluxo do RePO consome 77 features e classifica uma amostra pelo **erro de
reconstrução** de um autoencoder sob mascaramento aleatório.  Cada máscara binária zera 75 % das
features (mantém ~19 de 77) e produz um erro; o escore é a agregação de **N = 100** erros.  Em
produção isso significa **100 passagens pelo autoencoder por amostra** — 100× o custo de uma
única inferência.

O custo cresce linearmente com N e é o gargalo dominante do detector:

| modo | TPR macro (FPR 1 %) | TPR ponderada | máscaras/amostra |
|---|---|---|---|
| RePO | 0,2396 | 0,1581 | 100 |
| RePO+ | **0,2546** | **0,1873** | 100 |

*(`baseline_flow.json`, 144.182 amostras, 1 repetição de escore.)*

O `cost` dos JSONs registra o trabalho do experimento em três contadores: `masked_rows` (linhas
enviadas ao autoencoder, uma por amostra e máscara), `model_calls` (passagens para a frente, em
lotes de ~1.024 linhas) e `model_seconds` (tempo dentro do modelo).  O baseline enviou
**28.836.400** linhas em **28.162** chamadas (3,93 ms por chamada; 110,6 s no modelo; 309 s de
parede), porque pontua RePO e RePO+ em duas passagens; a cascata enviou **72.091.000** linhas em
**70.405** chamadas (3,77 ms; 265,2 s no modelo; 661 s de parede), porque roda 5 repetições.  Os
dois números **não são comparáveis como custo de produção**: eles medem o custo de *rodar o
experimento*.  O custo de uma decisão em produção é o número de máscaras que a regra usa —
100 no RePO+ e 60,8 na cascata — e é essa a métrica usada no resto do documento.  (Se a produção
repetir o escore 5 vezes, como na avaliação, os dois valores são multiplicados por 5; a razão
entre eles não muda.)

O RePO+ não usa a média: ele divide as 100 máscaras em **5 grupos de 20**, toma o **mínimo** de
cada grupo e **soma** os 5 mínimos.  A motivação é estatística: o mínimo por grupo é mais estável
que a média quando o erro tem cauda longa, e a soma de 5 mínimos preserva a escala.  O preço é
que o RePO+ continua precisando das 100 máscaras para calcular os 5 mínimos de 20.

### 2.2 Reduzir estaticamente (A) ≠ alocar dinamicamente (B)

A **direção A** responde "quantas máscaras bastam?" e troca N por um valor menor para *todas* as
amostras (por exemplo, 25).  É uma decisão de **hiperparâmetro**: o mesmo orçamento é pago por
uma amostra trivial e por uma amostra ambígua.  Os dados da curva de variância mostram o custo
dessa uniformidade: com 25 máscaras iid o TPR macro fica em 0,2509 e o ponderado em 0,1792
(contra 0,2560 / 0,1871 com 100).  Com 5 máscaras, 0,2439 / 0,1565.

A **direção B** faz outra pergunta: *"esta amostra precisa das 100 máscaras?"*.  A ideia é que a
maioria do tráfego é fácil — um prefixo pequeno de máscaras já produz um escore muito abaixo do
threshold — e que as amostras difíceis (ataques e benignos ambíguos) podem pagar o ensemble
completo.  O orçamento passa a ser uma **variável aleatória por amostra**, não uma constante.

Há uma diferença de garantia importante:

- em A, o escore de **toda** amostra muda quando N muda, inclusive o das amostras difíceis;
- em B, a amostra roteada é julgada pelo **mesmo RePO+ de 100 máscaras** do artigo.  A função de
  decisão na região difícil é idêntica à original; só a região fácil é resolvida mais barato.

---

## 3. Método didático

### 3.1 Máscara, escore, RePO e RePO+

- **Máscara** `m ∈ {0,1}^77`: vetor binário aplicado por multiplicação à amostra normalizada.
  `m_j = 0` zera a feature `j`; cada `m_j` é 0 com probabilidade 0,75, de forma independente
  (`IIDMaskPolicy`, `new_data_eval/masking.py`).  As máscaras são **estateless**: a semente de
  cada tripla `(amostra, fase, repetição)` é derivada por BLAKE2b, então o escore não depende de
  batch nem de ordem.
- **Erro de máscara** `e_i(m)`: MSE de reconstrução da amostra `i` sob a máscara `m`.
- **RePO (artigo / 1 máscara)** vs. **Média simples (`mean`, 100 máscaras)**: no artigo original o RePO utiliza 1 máscara por amostra; nos experimentos da melhoria (direção B sequencial), usa-se `s_i = (1/N) Σ_k e_i(m_k)` (média dos N=100 erros) como baseline comparativo da regra sequencial.
- **RePO+**: com N divisível por 5, `s_i = Σ_{g=1..5} min_{k ∈ grupo g} e_i(m_k)`.
  Com 100 máscaras são 5 grupos de 20.
- **Threshold** `τ`: calibrado nos escores benignos para um FPR alvo.  A implementação
  (`threshold_at_fpr`) usa a regra do artigo: o `ceil(n·fpr)`-ésimo maior escore benigno.
  O ponto de operação deste documento é **FPR = 1 %**.
- **TPR macro**: média não ponderada das TPRs das **11 classes de ataque** de
  `FLOW_ATTACKS` (Web Attack é excluído das médias, como no artigo).  **TPR ponderada**: média
  ponderada pelo número de amostras de cada classe (42.003 amostras de ataque).

### 3.2 O que é uma cascata

Uma cascata é um classificador em estágios com **custo crescente** e **regra de parada**:

1. o **estágio barato** calcula um escore com poucas máscaras (`n_cheap` = 5, 10 ou 25);
2. se esse escore é **muito baixo** (`< t_low`), a amostra é declarada benigna e o estágio caro
   **não roda**;
3. caso contrário, roda-se o **RePO+ completo** (100 máscaras) e aplica-se o threshold final `τ`.

O ponto sutil é que `t_low` **não é** o threshold de decisão.  Ele é calibrado para ser bem mais
baixo (na prática, 34× a 205× menor que `τ`, ver §7.1), de modo que só caem no atalho as amostras
cujo escore barato está longe da fronteira.  Um escore barato baixo é evidência forte de
benignidade; um escore barato alto não decide nada — apenas manda a amostra para o ensemble.

### 3.3 O que é a decisão sequencial

A decisão sequencial não tem dois estágios: ela **revela máscaras em blocos de 5** e para quando
o **intervalo de confiança de 95 %** da média exclui o threshold.

- Se o intervalo inteiro está **acima** de `τ` → ataque.
- Se o intervalo inteiro está **abaixo** de `τ` → benigno.
- Se o intervalo contém `τ` → revela mais 5 máscaras.
- No limite de 100 máscaras, decide pela média pontual.

É o análogo de um teste sequencial: amostras fáceis (média longe do threshold, variância baixa)
param cedo; amostras ambíguas pagam até 100.  O escore usado neste estudo de melhoria é a
**média aritmética das máscaras (chamada de `RePO` médio nos logs do estudo)**,
porque é a média que tem intervalo de confiança simples (ao contrário do RePO do
artigo original, que avalia apenas 1 máscara aleatória por amostra); o RePO+
(soma de mínimos) não tem esse estimador.

### 3.4 Diagramas

Cascata:

```
amostra x (77 features)
        │
        ▼
┌────────────────────────────────┐
│ ESTÁGIO BARATO                 │  custo = n_cheap (5, 10 ou 25)
│ RePO+ sobre as n_cheap         │
│ primeiras máscaras → s_cheap   │
└───────────────┬────────────────┘
                │
        s_cheap < t_low ?
          ┌─────┴─────┐
         sim          não
          │            │
     BENIGNO          ▼
     custo n_cheap   ┌──────────────────────────────┐
                     │ ESTÁGIO CARO                 │  custo = 100
                     │ RePO+ completo (100 máscaras)│
                     │ → s_full                     │
                     └──────────────┬───────────────┘
                                    │
                             s_full ≥ τ ?
                              ┌─────┴─────┐
                             sim          não
                          ATAQUE       BENIGNO
```

Decisão sequencial:

```
k = 5, 10, 15, ..., 100
     │
     ▼
m_k = média dos k erros ;  IC95% = m_k ± 1,96 · s_k / √k
     │
     ├── limite inferior ≥ τ  → ATAQUE,  para (custo k)
     ├── limite superior < τ  → BENIGNO, para (custo k)
     └── caso contrário       → k += 5
```

---

## 4. Algoritmo

Todo o mecanismo opera sobre a matriz de erros `E ∈ R^{n×100}` já calculada: o custo reportado é
o número de máscaras que a regra **usaria**, não uma re-execução do modelo
(`new_data_eval/decision.py`).

### 4.1 Escore barato e calibração de `t_low`

```
função ESCORE_BARATO(E, n_cheap):
    # o prefixo é reorganizado nos MESMOS 5 grupos do RePO+,
    # com n_cheap/5 máscaras por grupo
    G  ← 5
    k  ← n_cheap / G
    M  ← reshape(E[:, 0:n_cheap], (n_amostras, G, k))
    retorna soma(min(M, eixo=2), eixo=1)

função CALIBRA_T_LOW(escores_baratos_dos_ataques, orcamento_FN):
    se orcamento_FN == 0:
        retorna min(escores) − 1e-12        # nenhum ataque pode pular
    retorna quantil(escores, orcamento_FN)  # descarta no máximo essa fração
```

`t_low` é o **maior** valor que mantém a fração de ataques descartados dentro do orçamento.  O
orçamento é expresso em fração de **todas as 44.182 linhas de ataque** (as 11 classes mais Web
Attack, que entra no orçamento mesmo ficando fora da média macro).

Note que o escore barato **não é uma versão imparcial do escore completo**: com `n_cheap/5`
máscaras por grupo, cada mínimo é tomado sobre menos sorteios e é estocasticamente maior que o
mínimo de 20.  O escore barato é, portanto, viesado **para cima** em relação ao RePO+ de 100
máscaras.  Isso reforça a assimetria da regra: um escore barato baixo é evidência forte de
benignidade (o completo tenderia a ser ainda mais baixo), enquanto um escore barato alto é
inconclusivo — e por isso a amostra é roteada.

### 4.2 Threshold final e laço de decisão

```
função CASCATA(E, n_cheap, t_low, τ):
    s_cheap ← ESCORE_BARATO(E, n_cheap)
    roteada ← s_cheap ≥ t_low

    s ← cópia de s_cheap
    s[roteada] ← ESCORE_BARATO(E[roteada], 100)   # RePO+ completo
    máscaras ← onde(roteada, 100, n_cheap)
    decisão  ← s ≥ τ
    retorna (s, roteada, máscaras, decisão)

# calibração do threshold final, no conjunto benigno:
s_probe   ← CASCATA(E, n_cheap, t_low, τ = 0).s
τ         ← threshold_at_fpr(s_probe[benignos], fpr = 0.01)
```

Duas propriedades caem direto do código:

1. como `t_low ≪ τ`, toda amostra não roteada tem `s_cheap < t_low < τ` e é benigna **por
   construção** — o atalho nunca produz um positivo;
2. o threshold final é calibrado sobre o **escore cascateado** benigno.  Como os 1 % de benignos
   mais bem pontuados estão sempre roteados, `τ` coincide com o threshold do RePO+ completo
   (0,10372 nos dois casos, ver §6.1), e o ponto de operação é preservado.

Custo esperado por amostra:

```
máscaras/amostra = n_cheap + f_roteada · (100 − n_cheap)
f_roteada        = p_benigno · (1 − skip) + p_ataque · (1 − FN)
```

### 4.3 Decisão sequencial

```
função SEQUENCIAL(E, τ, bloco=5, k_min=5, k_max=100):
    pendente ← todas as amostras
    para k = k_min, k_min+bloco, ..., k_max:
        para i em pendente:
            m ← média(E[i, 0:k])
            s ← desvio_padrão(E[i, 0:k], ddof=1)
            L ← m − 1,96 · s / √k
            U ← m + 1,96 · s / √k
            se L ≥ τ:  decisão[i] ← ATAQUE;  máscaras[i] ← k; pendente −= i
            se U < τ:  decisão[i] ← BENIGNO; máscaras[i] ← k; pendente −= i
    para i ainda pendente:                      # chegou a 100 sem decidir
        decisão[i] ← média(E[i, 0:100]) ≥ τ
        máscaras[i] ← 100
```

A implementação real vetoriza o laço sobre as amostras e mantém `pendente` como máscara booleana.

### 4.4 Por que isso é política por amostra, não hiperparâmetro

- `n_cheap`, `t_low` e `τ` são **fixos**; o que varia é **quantas máscaras cada amostra consome**,
  decidido em tempo de inferência pelo escore barato.
- A amostra difícil **não é truncada**: ela recebe o ensemble completo e é julgada pela função de
  decisão original.  A cascata não é "RePO+ com 10 máscaras"; é "RePO+ com 100 máscaras para quem
  precisa, 10 para quem não precisa".
- O custo é uma variável aleatória que depende da distribuição de escores da carga — por isso a
  economia precisa ser reportada com a prevalência (§8), e não como um número único.
- A contrapartida é que o atalho vira **superfície de ataque**: quem consegue empurrar o escore
  barato para baixo de `t_low` nunca enfrenta o ensemble.  É exatamente isso que o protocolo de
  §7 mede.

---

## 5. Protocolo experimental

**Amostra.** CICIDS2017, quatro dias de teste (Tuesday–Friday), 77 features de fluxo.  Amostragem
uniforme por classe (reservatório pela menor prioridade aleatória, semente 2019), com teto de
100.000 benignos e 5.000 por ataque: **144.182 amostras** = 100.000 benignos + 44.182 ataques.
As classes com menos linhas entram inteiras (Heartbleed 11, Infiltration 36, Bot 1.956, Web Attack
2.179); as outras oito têm 5.000 cada.

**Modelo.** `models/flw_model` (SavedModel do autoencoder do artigo), normalização com
`train_min`/`train_max` do próprio conjunto.  Máscaras iid com p = 0,75 de mascaramento.

**Repetições.** O `cascade_flow.json` usa **5 repetições de escore** (`score_repeats = 5`): cinco
conjuntos independentes de 100 máscaras por amostra; cada métrica é a média das cinco.  O
`baseline_flow.json` usa **1 repetição**.  Essa é a única diferença metodológica entre a
referência interna da cascata e o baseline do artigo, e ela explica as pequenas discrepâncias de
§6 (o desvio entre repetições do TPR macro do RePO+ com 100 máscaras é ≈ 0,0011; ver
`variance_flow.json`).

**Threshold final.** FPR de 1 % nos 100.000 benignos, recalibrado em cada repetição e em cada
configuração.

**Threshold de ataque (protocolo adversarial).** Ponto de operação **≈ FPR 10 %**, fixo e igual
para a amostra normal e para a vítima.  O protocolo aparece em duas variantes: na direção A,
`attack_threshold = 0,8 × threshold real` (um pouco mais permissivo que 10 % de FPR;
`mask_experiments/adversarial.py`, `ATTACK_FACTOR = 0.8`); no ataque à cascata, calibração
direta no quantil de 10 % (`cascade_attack.py`, `--fpr 0.10`).  O valor gravado nos JSONs é
**0,015859**, idêntico ao threshold de RePO+ a 10 % do baseline.  Consequência importante: **as
TPRs adversariais não são as TPRs de produção a 1 %** — são as de um ponto bem mais permissivo.

**Métricas.**

- normais: TPR macro/ponderada, máscaras por amostra na mistura 99/1, taxa de benignos que pulam
  o ensemble, FN do estágio barato, FPR, concordância com o detector completo;
- adversariais: `normal_tpr`, `victim_tpr`, `bypass_rate` (fração de vítimas que não chegam ao
  ensemble), `fooled` (detectadas no normal e não detectadas na vítima), `already_benign`
  (falsos negativos que já existiam antes do ataque), `kinds` (evaded/backup/no_change/
  not_fooled) e `masks_evaluated` (custo do **ataque**, não da detecção).

**Ataques.** 11 classes: FTP-Patator, SSH-Patator, DoS slowloris, DoS Slowhttptest, DoS Hulk, DoS
GoldenEye, Heartbleed, Infiltration, Bot, PortScan, DDoS.  1.000 amostras por ataque, exceto
Heartbleed (11) e Infiltration (36), que são todas as linhas disponíveis → **9.047 amostras**
atacadas.  O atacante otimiza um modificador `alpha` com a projeção de viabilidade do ataque de
referência (`new_data_eval/attack.py`), em 3 camadas de máscara fixa × 4 taxas de aprendizado ×
(30, 40, 50, 60) iterações.

**Execução.** `cascade`: job 30233; `cascade_attack`: job 30234; a rodada de 11 ataques × 1.000
amostras: job 31954 (`chain_manifest.tsv`, `chain_manifest_resume.tsv`).  Tempo de parede
registrado: 661 s para a cascata e 24,6 h somadas para os 11 ataques.

---

## 6. Resultados normais

Referências: as duas primeiras linhas são o **baseline do artigo** (`baseline_flow.json`, 1
repetição); a terceira é o **`full_repo_plus` do `cascade_flow.json`** (5 repetições), que é a
referência interna usada em todas as comparações de custo.  `ganho` = redução relativa de
máscaras/amostra contra 100.

| regra | TPR macro | TPR pond. | máscaras/amostra | ganho | benignos pulados | FN do estágio barato | FPR | concordância |
|---|---|---|---|---|---|---|---|---|
| **Média simples** (baseline, 100) | 0,2396 | 0,1581 | 100,00 | 0,0 % | — | — | 1,00 % | — |
| **RePO+** (artigo, 100) | 0,2546 | 0,1873 | 100,00 | 0,0 % | — | — | 1,00 % | — |
| RePO+ referência interna (5 rep.) | 0,2560 | 0,1871 | 100,00 | 0,0 % | 0,00 % | 0,00 % | 1,00 % | — |
| cascata cheap5, FN 0 | 0,2560 | 0,1871 | 72,66 | 27,3 % | 29,07 % | 0,00 % | 1,00 % | — |
| cascata cheap5, FN 0,1 % | 0,2560 | 0,1871 | 67,42 | 32,6 % | 34,64 % | 0,10 % | 1,00 % | — |
| cascata cheap5, FN 1 % | 0,2560 | 0,1871 | 64,83 | 35,2 % | 37,39 % | 1,00 % | 1,00 % | — |
| cascata cheap10, FN 0 | 0,2560 | 0,1871 | 66,37 | 33,6 % | 37,74 % | 0,00 % | 1,00 % | — |
| cascata cheap10, FN 0,1 % | 0,2560 | 0,1871 | 64,04 | 36,0 % | 40,36 % | 0,10 % | 1,00 % | — |
| **cascata cheap10, FN 1 %** | **0,2560** | **0,1871** | **60,80** | **39,2 %** | **43,98 %** | **1,00 %** | **1,00 %** | — |
| cascata cheap25, FN 0 | 0,2560 | 0,1871 | 80,81 | 19,2 % | 25,85 % | 0,00 % | 1,00 % | — |
| cascata cheap25, FN 0,1 % | 0,2560 | 0,1871 | 69,83 | 30,2 % | 40,63 % | 0,10 % | 1,00 % | — |
| cascata cheap25, FN 1 % | 0,2560 | 0,1871 | 68,84 | 31,2 % | 41,96 % | 1,00 % | 1,00 % | — |
| **RePO sequencial** | 0,2438 | 0,1577 | 5,32 | 94,7 % | — | — | 0,99 % | 99,93 % |

### 6.1 Leitura

1. **A TPR é preservada, não melhorada.**  As nove configurações de cascata têm TPR macro
   0,2559555 e ponderada 0,1870866 — **idênticas** ao RePO+ completo da mesma rodada, até a sexta
   casa.  O motivo está em §4.2: o threshold final é o mesmo e nenhuma amostra descartada teria
   sido detectada (o limite de pior caso do orçamento de 1 % é 1 ponto de TPR macro; o valor
   observado é 0).
2. **O melhor ponto da grade é cheap10 + FN 1 %**, com 60,80 máscaras.  Não é o `n_cheap` que
   minimiza o custo: com 5 máscaras o escore barato é mais ruidoso, o `t_low` calibrado sobe e
   **menos** benignos pulam (37,4 % contra 44,0 %).  Com 25 máscaras o estágio barato já custa
   caro e o `t_low` fica tão baixo (0,000506) que só 42,0 % pulam.  O ótimo é um compromisso entre
   custo do prefixo e seletividade do atalho, não um extremo.
3. **O orçamento de FN é consumido exatamente.**  1 % de 44.182 = 442 ataques descartados; o
   valor medido é 1,0004 % (a granularidade de 1/44.182 explica o desvio de 0,0004 pp).  Com
   0,1 %, o medido é 0,1019 % (45 ataques).
4. **O FPR é 1,00 % por construção** em todas as cascatas: o threshold é calibrado no escore
   cascateado benigno.  O sequencial mede 0,9916 % — ligeiramente abaixo porque o intervalo de
   confiança às vezes decide "benigno" antes de a média cruzar o threshold.
5. **O sequencial é outro trade-off.**  Ele custa 5,32 máscaras (−94,7 %) e concorda em 99,93 %
   com a decisão do RePO de 100 máscaras, mas parte do **RePO** (média), não do RePO+: sua TPR
   macro é 0,2438 e a ponderada 0,1577, contra 0,2560/0,1871 do RePO+.  A coluna de TPR é a do
   escore pontual do RePO (como reportado pelo código); a regra sequencial decide o mesmo em
   99,93 % dos casos, então a TPR realizada fica a uma fração de ponto percentual desse valor.
6. **Tamanho do efeito versus ruído.**  A diferença entre o RePO+ do artigo (0,2546) e a
   referência interna (0,2560) é de 0,0013 — compatível com o desvio entre repetições
   (≈ 0,0011).  Não é uma mudança de comportamento do detector; é a média de cinco sorteios
   contra um.

---

## 7. Resultados adversariais

Protocolo: o atacante **conhece o detector** e otimiza o **estágio barato** até o escore de
`n_cheap` máscaras cair abaixo de `t_low`.  Se conseguir, a amostra é declarada benigna e o
ensemble não participa da decisão.  O ponto de operação é ≈ FPR 10 % e é o mesmo para normal e
vítima.  Cada célula é `TPR normal → TPR vítima (bypass)`.

| ataque | n | cheap5 | cheap10 | cheap25 |
|---|---|---|---|---|
| FTP-Patator | 1.000 | 0,504 → 0,504 (0,0 %) | 0,504 → 0,499 (0,0 %) | 0,504 → 0,502 (0,1 %) |
| SSH-Patator | 1.000 | 0,007 → 0,006 (**5,3 %**) | 0,007 → 0,006 (**2,0 %**) | 0,007 → 0,006 (0,0 %) |
| DoS slowloris | 1.000 | 0,665 → 0,664 (0,0 %) | 0,665 → 0,665 (0,0 %) | 0,665 → 0,665 (0,0 %) |
| DoS Slowhttptest | 1.000 | 0,929 → 0,932 (0,0 %) | 0,929 → 0,932 (0,0 %) | 0,929 → 0,932 (0,0 %) |
| DoS Hulk | 1.000 | 0,991 → 0,991 (0,0 %) | 0,991 → 0,991 (0,0 %) | 0,991 → 0,991 (0,0 %) |
| DoS GoldenEye | 1.000 | 0,719 → 0,719 (0,0 %) | 0,719 → 0,719 (0,0 %) | 0,719 → 0,719 (0,0 %) |
| Heartbleed | 11 | 1,000 → 1,000 (0,0 %) | 1,000 → 1,000 (0,0 %) | 1,000 → 1,000 (0,0 %) |
| Infiltration | 36 | 0,694 → 0,667 (0,0 %) | 0,694 → 0,667 (**2,8 %**) | 0,694 → 0,667 (0,0 %) |
| Bot | 1.000 | 0,029 → 0,029 (0,7 %) | 0,029 → 0,029 (0,2 %) | 0,029 → 0,029 (0,1 %) |
| PortScan | 1.000 | 0,112 → 0,117 (0,0 %) | 0,112 → 0,117 (0,0 %) | 0,112 → 0,117 (0,0 %) |
| DDoS | 1.000 | 0,640 → 0,641 (0,0 %) | 0,640 → 0,641 (0,0 %) | 0,640 → 0,641 (0,0 %) |
| **agregado** | **9.047** | **0,663 %** | **0,254 %** | **0,022 %** |

*(bypass agregado = Σ bypass·n / Σ n.  IC95 % de Wilson: 0,516–0,853 % para cheap5;
0,169–0,381 % para cheap10; 0,006–0,081 % para cheap25.)*

### 7.1 O que o bypass mede (e o que ele não mede)

`bypass_rate` é a fração de vítimas com `s_cheap < t_low`, ou seja, que **pulam o ensemble**.  Como
`t_low` é ≈ 34× (cheap5), 60× (cheap10) ou 205× (cheap25) menor que o threshold de decisão, o alvo
do ataque é **mais estrito** do que evadir a decisão: todo escore abaixo de `t_low` também está
abaixo de `τ`, de modo que todo bypass é, por construção, também uma evasão da decisão.  O que se
mede é, portanto, um **limite inferior** da evasão.  Em contrapartida, baixo bypass **não**
significa que a amostra foi detectada — significa apenas que o ensemble foi efetivamente
consultado e a decisão voltou para ele.

Duas ressalvas de leitura:

- **`bypass` não é `fooled`.**  `fooled` conta detecções perdidas; `bypass` conta pulos.  PortScan e
  Slowhttptest têm `fooled` = 11 com `bypass` = 0: a cascata roteou corretamente e a perda ocorreu
  no escore completo, não no atalho.  Nos dois casos a TPR da vítima fica **acima** da normal
  (0,112 → 0,117 e 0,929 → 0,932), o que é ruído de reamostragem de máscaras: a vítima é pontuada
  com máscaras de outra fase (`PHASE_VICTIM`) e, com n = 1.000, o desvio binomial em p ≈ 0,1 já é
  ≈ 1 ponto percentual.
- **`already_benign` importa.**  Em SSH-Patator, 993/1.000 amostras já eram falsos negativos antes
  de qualquer otimização; em Bot, 971; em PortScan, 888.  O bypass medido nesses casos é
  majoritariamente "o detector já era cego", não "o atalho foi quebrado".

### 7.2 As duas exceções, honestamente

**SSH-Patator (TPR normal 0,007).**  É a exceção real: bypass de 5,3 % com 5 máscaras, 2,0 % com
10 e 0 % com 25.  Mas o número absoluto engana.  No ponto de operação de 10 % de FPR o detector
marca apenas **7 de 1.000** amostras dessa classe; `fooled` = 1 nas três configurações
(`conditional_drop` = 14,3 %, isto é, 1 de 7).  Como toda amostra que pula é, por definição, não
detectada na vítima, no máximo `fooled` = 1 dos 53 pulos de cheap5 pode ter sido uma detecção
perdida; **≥ 52 eram falsos negativos que já existiam**.  Conclusão defensável: o bypass do SSH
mede a fragilidade do detector para essa classe, não a fragilidade do estágio barato.  O SSH é,
na prática, um ponto fora do envelope de operação do modelo — a 1 % de FPR sua TPR no baseline é
0,0002 (1 em 5.000).

**Infiltration (n = 36, bypass 2,8 % com 10 máscaras).**  É **um** evento em 36 amostras, e esse
evento foi uma detecção perdida (`fooled` = 1; TPR 0,694 → 0,667).  O intervalo de Wilson de 95 %
para 1/36 vai de 0,5 % a 14,2 %: o dado não permite afirmar uma taxa.  Com 5 e com 25 máscaras o
bypass é 0.  O correto é registrar "1 de 36" e não tratar 2,8 % como estimativa.

**Se excluirmos o SSH-Patator** — que, como visto, mede cegueira pré-existente — o pior bypass é
0,7 % (Bot, 7/1.000 com cheap5) e o agregado fica em **0,09 % (7/8.047)** com cheap5,
**0,04 % (3/8.047)** com cheap10 e **0,02 % (2/8.047)** com cheap25.  No cheap10, um dos 3 é o
único evento de Infiltration.

### 7.3 Custo do ataque

`masks_evaluated` é a média de linhas mascaradas avaliadas **pelo otimizador**, não pela decisão:
≈ 5.405 (cheap5), 10.810 (cheap10) e 27.025 (cheap25), o equivalente a ~1.081 avaliações de escore
por amostra (540 iterações de otimização × 2 avaliações, mais a avaliação inicial).  Quando o
ataque encontra a evasão cedo, o custo cai (Bot cheap5: 5.224; SSH cheap5: 4.935); quando não
encontra, ele gasta o orçamento inteiro.  Esses números **não** entram no custo de detecção de
§6 — são o custo de montar o ataque.

---

## 8. Análise custo × robustez

### 8.1 A fórmula e uma conta completa

```
máscaras/amostra = n_cheap + f_roteada · (100 − n_cheap)
f_roteada        = p_benigno · (1 − skip) + p_ataque · (1 − FN)
```

Para cheap10, FN 1 % (o ponto de §6):

```
skip = 0,439832        (43,98 % dos benignos pulam)
FN   = 0,010004        (1,0004 % dos ataques pulam)
p_benigno = 0,99 ; p_ataque = 0,01

f_roteada = 0,99 · (1 − 0,439832) + 0,01 · (1 − 0,010004)
          = 0,99 · 0,560168 + 0,01 · 0,989996
          = 0,554566 + 0,009900
          = 0,564466

máscaras/amostra = 10 + 0,564466 · 90 = 10 + 50,802 = 60,802   ✔
```

O valor bate com `masks_used_mix` do JSON (60,801965) porque `cascade.py` usa exatamente
`BENIGN_FRACTION = 0.99`.  A mesma conta reproduz as outras oito configurações dentro de 10⁻⁴
(§12).

### 8.2 Sensibilidade à prevalência

A economia não é uma propriedade do detector: é uma propriedade do **detector sob uma carga**.
Como só benignos podem pular, quanto mais ataque no tráfego, menor o ganho.

| p(benigno) / p(ataque) | f_roteada | máscaras/amostra | ganho vs 100 |
|---|---|---|---|
| 100 % / 0 % | 0,560168 | **60,42** | 39,6 % |
| 99 % / 1 % (mistura de avaliação) | 0,564466 | **60,80** | 39,2 % |
| 90 % / 10 % | 0,603151 | **64,28** | 35,7 % |
| 50 % / 50 % | 0,775082 | **79,76** | 20,2 % |
| 10 % / 90 % | 0,947013 | **95,23** | 4,8 % |
| 0 % / 100 % | 0,989996 | **99,10** | 0,9 % |

Leitura: o número de manchete (60,8) vale para a mistura 99/1 do protocolo.  Numa rede com 50 %
de ataque — um cenário irrealista, mas útil como limite — o ganho cai para 20 %; num fluxo
inteiramente hostil, a cascata praticamente não economiza, porque o orçamento de 1 % de FN impede
que ataques pulem.  **A conclusão de custo deve ser sempre acompanhada da prevalência.**

### 8.3 Custo × robustez

| regra | máscaras/amostra | TPR macro | bypass agregado no atalho (FN 1 %) | observação |
|---|---|---|---|---|
| RePO+ 100 | 100,00 | 0,2560 | — | referência |
| cascata cheap10, FN 0 | 66,37 | 0,2560 | não medido | zero FN no atalho |
| cascata cheap10, FN 1 % | **60,80** | 0,2560 | 0,25 % | melhor custo da grade |
| cascata cheap25, FN 1 % | 68,84 | 0,2560 | 0,02 % | mais robusto, mais caro |
| RePO sequencial | **5,32** | 0,2438 | não avaliado | sem avaliação adversarial |

O padrão é claro: **a cascata compra custo com uma superfície de ataque nova, e essa superfície é
pequena no protocolo medido**.  O sequencial compra muito mais custo, mas troca de detector
(RePO em vez de RePO+) e não foi atacado — a comparação direta entre os dois não está fechada.

---

## 9. Ameaças à validade

1. **Ataques a 1.000 amostras.**  É suficiente para estimar bypass de 0,2–0,7 % com IC estreito,
   mas não para eventos raros.  Para 0/1.000 o limite superior de Wilson é 0,38 %; para 1/1.000,
   0,56 %.  Ou seja, "bypass ≈ 0" significa "≤ 0,4 % com 95 % de confiança" por classe, não zero
   absoluto.
2. **Heartbleed (n = 11) e Infiltration (n = 36).**  Nenhuma taxa dessas classes é estimável com
   precisão; 0/11 tem IC superior de 25,9 %.  Elas entram na tabela por completude, não como
   evidência.
3. **O orçamento de FN é real.**  1 % dos ataques **não** enfrenta o ensemble.  Na amostra, nenhum
   deles teria sido detectado (a TPR não se moveu), mas o limite de pior caso é 1 ponto de TPR
   macro.  Além disso, `t_low` é calibrado **nos mesmos ataques que depois são avaliados**: a
   garantia de FN ≤ 1 % é in-sample.  Num conjunto de ataques diferente, o orçamento pode ser
   consumido por amostras que o detector detectaria.
4. **O threshold de ataque é permissivo (FPR ≈ 10 %).**  As TPRs de §7 (FTP 0,504; SSH 0,007;
   PortScan 0,112) não são as de produção a 1 % (FTP 0,0000; SSH 0,0002; PortScan 0,0000 no
   baseline).  O estudo mede robustez no regime em que o detector ainda vê essas classes; a 1 % de
   FPR a maior parte delas está fora do envelope.
5. **O sequencial não teve avaliação adversarial.**  Um atacante que conhece a regra pode tentar
   inflar a variância para forçar 100 máscaras (negação de serviço) ou empurrar a média para o
   lado benigno com poucas máscaras.  Nada disso foi medido; a concordância de 99,93 % é com o
   tráfego limpo.
6. **B não foi avaliada em pacotes.**  `packet_normal.json` só compara políticas de máscara
   (iid/balanced/covering) em 1.994 janelas normais; não há cascata nem ataque de pacotes.
7. **Ataque e defesa compartilham o esquema de máscaras.**  As máscaras da fase de ataque e as da
   fase de vítima são independentes, mas ambas são derivadas deterministicamente do índice global
   da amostra.  Um atacante que não conheça esse esquema pode ser menos eficaz; um que conheça tem
   a vantagem medida aqui.
8. **Ruído de reamostragem.**  `victim_tpr` usa máscaras novas; diferenças de 1–2 amostras por
   classe (≈ 0,1–0,2 pp) não devem ser interpretadas como efeito do ataque.  Onde a TPR da vítima
   sobe (PortScan, Slowhttptest, DDoS), o mais provável é ruído, não "o ataque ajudou o detector".
9. **Uma única arquitetura e um único dataset.**  Tudo aqui é CICIDS2017-fluxo com o autoencoder
   do artigo.  A forma da cauda do escore barato — que determina `skip` e, portanto, toda a
   economia — pode mudar em outro modelo ou outro tráfego.

---

## 10. Conclusão defensável

A direção B entrega o que promete, com uma frase precisa: **a cascata cheap-first preserva a
decisão do RePO+ de 100 máscaras nas amostras que importam e reduz o custo médio para 60,8
máscaras/amostra (−39,2 %) na mistura 99/1, com 43,98 % dos benignos resolvidos por 10 máscaras.**

O que se pode afirmar:

- A TPR normal é **idêntica** à do RePO+ completo na mesma rodada (0,2560 macro / 0,1871
  ponderada), com o mesmo FPR de 1 % e o mesmo threshold final.  Não há ganho de acurácia; há
  ganho de custo.
- O estágio barato é **resistente no protocolo medido**: bypass agregado de 0,25 % com 10 máscaras
  (IC95 % 0,17–0,38 %) e 0,02 % com 25; fora SSH-Patator, 0,04 %.
- As exceções são explicáveis e não sustentam uma taxa: SSH tem TPR 0,007 no ponto de operação
  (o bypass mede cegueira pré-existente); Infiltration tem 1 evento em 36.
- A economia é **dependente da carga**: 60,4 com 100 % de benignos, 79,8 com 50/50, 99,1 com 100 %
  de ataques.

O que **não** se pode afirmar:

- que a cascata é mais robusta que o RePO+ — ela adiciona um caminho de decisão que não existia e
  o ataque a esse caminho é possível em princípio;
- que 60,8 é um número universal — ele é a média na mistura 99/1 com `t_low` calibrado nos
  próprios ataques avaliados;
- que a cascata é a melhor direção — o sequencial chega a 5,32 máscaras, mas troca RePO+ por RePO,
  perde 1,2 pp de macro e não foi atacado.  A comparação final entre B1 (cascata) e B2
  (sequencial) exige o ataque ao sequencial e, idealmente, uma versão sequencial do RePO+.

**Recomendação de engenharia.**  Se o objetivo é reduzir custo sem tocar na decisão, `cheap10` com
FN de 1 % é o ponto defensável da grade: é o menor custo medido, mantém a TPR e mantém bypass
agregado abaixo de 0,3 %.  Se o ambiente tem prevalência de ataque muito maior que 1 %, o ganho
encolhe e a decisão deve ser reavaliada com a fórmula de §8.1.

---

## 11. Reprodutibilidade

Tudo roda a partir de `code/` (o pacote é `mask_experiments`; os caminhos de dados e modelo têm
defaults para o repositório):

```bash
cd code

# 1) cascata + sequencial (5 repetições, 144.182 amostras)
python -m mask_experiments.cascade \
  --data-dir ../data --model-dir ../models/flw_model \
  --max-benign 100000 --max-per-attack 5000 \
  --n-masks 100 --repeats 5 \
  --cheap 5,10,25 --fn-rates 0.0,0.001,0.01 \
  --batch-size 1024 --seed 2019 \
  --out ../results/mask_efficiency/cascade_flow.json

# 2) ataque ao estágio barato (uma chamada por ataque)
python -m mask_experiments.cascade_attack \
  --data-dir ../data --model-dir ../models/flw_model \
  --max-benign 100000 --max-per-attack 5000 --per-attack 1000 \
  --attacks "FTP-Patator" \
  --cheap 5,10,25 --fn-rate 0.01 --fpr 0.10 \
  --batch-size 1024 --seed 2019 \
  --out ../results/mask_efficiency/cascade_attack/FTP-Patator.json
```

No cluster, os mesmos comandos são emitidos por `cluster/run_mask_study.sh` via
`cluster/slurm/job.sh` (`mask_cascade` e `mask_cascade_attack`); o array de ataques usa
`ATTACK_LIST_FILE` + `SLURM_ARRAY_TASK_ID`.  Jobs: **30233** (cascata), **30234** (ataque à
cascata, piloto), **31954** (11 ataques × 1.000).  Sementes: 2019 no escore e nas máscaras; o
sorteio das amostras atacadas usa o gerador `seed + 4 = 2023`.

Artefatos esperados: `cascade_flow.json` (chaves `full_repo_plus`, `cascade`, `sequential_repo`,
`cost`, `dataset`) e `cascade_attack/<ATAQUE>.json` (chaves `configs` e `attacks`), no schema
`mask_exp_v1`.

---

## 12. Checagem de números

Três verificações independentes, recalculadas a partir dos JSONs:

**(a) Custo da melhor configuração.**  `cascade_flow.json` → `cascade.cheap10_fn0.01`:
`masks_used_mix = 60.801965133348425`, `benign_skip_rate = 0.439832`,
`cheap_false_negative_rate = 0.010004074057308406`.  Recalculando
`10 + [0,99·(1−0.439832) + 0,01·(1−0.010004)]·90 = 60.801965` → confere; ganho
`(100 − 60,801965)/100 = 39,198 %` → **−39,2 %**.

**(b) TPR preservada.**  `full_repo_plus.tpr_macro_mean = 0.2559555059800459` e
`cascade.*.tpr_macro_mean = 0.2559555059800459` nas nove configurações; idem para a ponderada
(0.18708663666880937).  Diferença `0,000000` → a cascata **não altera** a TPR nesta amostra.

**(c) Bypass agregado e orçamento de FN.**  Somando `bypass_rate × n` sobre os 11 JSONs:
cheap5 = 60/9.047 = 0,663 %; cheap10 = 23/9.047 = 0,254 %; cheap25 = 2/9.047 = 0,022 %.
Conferindo o denominador: 8 classes × 1.000 + Bot (1.000) + Infiltration (36) + Heartbleed (11) =
**9.047**.  O orçamento de 1 % sobre 44.182 ataques é 442 amostras; o medido é
`0,010004 × 44.182 = 442` → confere.  A conta de prevalência 50/50
(`10 + [0,5·0,560168 + 0,5·0,989996]·90 = 79,757`) também foi refeita e bate com a tabela de §8.2.

*(Conferência adicional: `144.182 × 100 × 5 = 72.091.000`, exatamente o `cost.masked_rows` do
`cascade_flow.json`; `144.182 × 100 × 2 = 28.836.400`, o `cost.masked_rows` do baseline, que
pontua os dois modos separadamente.)*
