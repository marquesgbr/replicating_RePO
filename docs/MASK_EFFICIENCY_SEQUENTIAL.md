# Decisão sequencial sob ataque adversarial

> Documento companheiro de `MASK_EFFICIENCY_CASCADE.md`. Aquele consolida a
> cascata *cheap-first* e a decisão sequencial no cenário **normal**; este mede
> o que acontece quando o atacante mira a **regra de parada**.

## 1. Resumo executivo

A decisão sequencial gasta, no cenário normal do artigo (FPR 1 %),
**5,32 máscaras por amostra** (−94,7 % contra o RePO+ de 100) com 99,93 % de
concordância. Sob ataque adaptativo à regra de parada, esse mesmo mecanismo é
**frágil**: nos ataques com TPR normal relevante, a regra perde de 57 % a 97 %
da detecção.

O ponto central do resultado não é a parada antecipada, é o **escore**: as
vítimas que enganam o primeiro bloco continuam enganando o RePO médio de 100
máscaras (`transfer_full100_tpr ≈ victim_tpr` em todas as linhas), mas são
pegas pelo RePO+ de 100 máscaras (`repo_plus_full100_tpr`) na maioria dos casos.
O gargalo é o escore médio, não o IC.

| ataque | normal | vítima | queda | parada no 1º bloco | RePO+ (mesmas vítimas) |
|---|---|---|---|---|---|
| DoS Hulk | 0,657 | 0,017 | −97,4 % | 97,1 % | 0,991 |
| DoS GoldenEye | 0,399 | 0,056 | −86,0 % | 83,9 % | 0,719 |
| DoS Slowhttptest | 0,405 | 0,068 | −83,2 % | 60,2 % | 0,938 |
| DoS slowloris | 0,347 | 0,118 | −66,0 % | 61,4 % | 0,660 |
| PortScan | 0,997 | 0,432 | −56,7 % | 16,0 % | 0,106 |
| DDoS | 0,681 | 0,611 | −10,3 % | 19,8 % | 0,667 |

Macro nos 9 ataques com n = 1.000: **0,392 → 0,151 (−61,5 %)**.
Macro nos 11 ataques (convenção do estudo, incluindo Heartbleed e
Infiltration): **0,480 → 0,275 (−42,7 %)**.

## 2. O que é atacável na regra sequencial

A regra revela máscaras em blocos de 5 e, a cada bloco, calcula o intervalo de
confiança de 95 % da média dos erros. Se o limite superior fica abaixo do
threshold `τ`, a amostra é benigna **naquele bloco**; se o inferior fica acima,
é ataque; se o intervalo contém `τ`, gasta mais máscaras.

Isso cria duas superfícies distintas:

1. **o valor do escore** — empurrar a média para baixo de `τ`, que é o ataque
   clássico do RePO (o mesmo da direção A);
2. **a parada** — fazer o *primeiro* bloco já parecer benigno, o que exige
   `média + z·erro-padrão < τ`, um alvo **mais difícil** que o escore pontual.

O experimento foi desenhado para o atacante mais forte (2) e mede as duas coisas
separadamente: `early_stop_rate` (a parada) e `victim_tpr` (a decisão final).
A comparação das duas com `transfer_full100_tpr` é o que separa "a parada é o
problema" de "o escore é o problema".

## 3. O ataque

### 3.1 Objetivo

O atacante otimiza a perturbação `α` minimizando, no bloco de `b` máscaras,

```
objetivo_ci(e) = Σ e_j + z · √(b · var(e)),     var com ddof = 1
               = b · (média(e) + z · erro-padrão(e))
```

que é `b` vezes o limite superior do intervalo da regra. O ataque declara
sucesso quando o escore **médio** do bloco cruza `τ` — a mesma estatística com
que a regra decide, e a razão de `FlowAttack` ganhar o parâmetro
`score_mode="repo"` (o agregado padrão, RePO+, coincide com `b × média` só
quando `b = 5`, em que cada grupo tem uma máscara; usar RePO+ para julgar um
ataque de bloco 10 ou 25 compararia estatísticas diferentes).

O objetivo clássico (`loss="sum"`, `Σ e_j`) também está disponível; no piloto de
20 amostras os dois produziram a mesma distribuição de desfechos e o mesmo
custo, então o estudo usa `ci`, que é o atacante mais forte.

### 3.2 Gradiente

A cada passo o atacante sorteia um bloco novo de `b` máscaras — mesma derivação
estatística `(seed, sample_index, iteração)` do resto do estudo —, calcula os
erros por máscara e aplica um passo de Adam na perda acima. O laço externo é o
do `FlowAttack`: 4 taxas de aprendizado × (30, 40, 50, 60) iterações, em três
máscaras de projeção fixas — **o mesmo orçamento** do ataque da cascata e da
direção A.

### 3.3 Avaliação da vítima

A decisão da vítima não é o critério do atacante: a amostra perturbada é
reavaliada pela regra real, com 100 máscaras novas (fase `victim`), e só então
`sequential_repo_decisions` decide. Isso permite separar

- `early_stop_rate` — fração de vítimas descartadas como benignas **no primeiro
  bloco** (o alvo do ataque);
- `victim_tpr` — decisão final da regra, que pode continuar revelando máscaras;
- `transfer_full100_tpr` — a mesma vítima julgada pelo RePO de média completa;
- `repo_plus_full100_tpr` — a mesma vítima julgada pelo RePO+ de 100 máscaras.

## 4. Protocolo

- 11 ataques do CICIDS2017 (fluxo), **1.000 amostras por ataque**; Heartbleed
  (n = 11) e Infiltration (n = 36) usam todas as disponíveis.
- 100.000 benignos; `seed = 2019`; máscaras `iid`.
- `τ` = threshold que produz **10 % de FPR** nos benignos — o mesmo protocolo do
  `cascade_attack`, para que as tabelas sejam comparáveis. Threshold de ataque =
  o mesmo `τ` (nada é relaxado para a vítima).
- Bloco 5 (headline); `z = 1,96`; `max_masks = 100`.
- Fases `attack` e `victim` com sementes de máscara independentes, como em todo
  o estudo.

## 5. Resultados normais (bloco 5, FPR 10 %)

| métrica | valor |
|---|---|
| `τ` | 0,01630 |
| FPR realizado | 9,72 % |
| TPR macro (11 ataques) | 0,4799 |
| TPR macro (9 ataques com n = 1.000) | 0,3918 |
| máscaras/amostra — benigno | 8,05 |
| máscaras/amostra — ataque | 13,55 |

O custo é maior que os 5,32 do cenário normal do artigo porque o protocolo de
ataque usa **FPR 10 %**: com `τ` mais baixo, uma amostra benigna precisa de mais
máscaras para que o limite superior do IC caia abaixo dele. A ordem de grandeza
da economia se mantém (−92 % contra 100).

## 6. Resultados adversariais (bloco 5, 1.000/ataque)

| ataque | n | normal | vítima | queda | parada 1º bloco | máscaras da vítima | RePO 100 | RePO+ 100 |
|---|---|---|---|---|---|---|---|---|
| Bot | 1.000 | 0,018 | 0,012 | −33,3 % | 82,1 % | 6,1 | 0,012 | 0,029 |
| DDoS | 1.000 | 0,681 | 0,611 | −10,3 % | 19,8 % | 22,9 | 0,608 | 0,667 |
| DoS GoldenEye | 1.000 | 0,399 | 0,056 | −86,0 % | 83,9 % | 7,2 | 0,056 | 0,719 |
| DoS Hulk | 1.000 | 0,657 | 0,017 | −97,4 % | 97,1 % | 5,4 | 0,018 | 0,991 |
| DoS Slowhttptest | 1.000 | 0,405 | 0,068 | −83,2 % | 60,2 % | 12,4 | 0,074 | 0,938 |
| DoS slowloris | 1.000 | 0,347 | 0,118 | −66,0 % | 61,4 % | 15,7 | 0,129 | 0,660 |
| FTP-Patator | 1.000 | 0,020 | 0,043 | ruído¹ | 61,7 % | 20,9 | 0,041 | 0,503 |
| Heartbleed | 11 | 1,000 | 1,000 | 0,0 % | 0,0 % | 5,0 | 1,000 | 1,000 |
| Infiltration | 36 | 0,750 | 0,667 | −11,1 % | 27,8 % | 11,2 | 0,750 | 0,667 |
| PortScan | 1.000 | 0,997 | 0,432 | −56,7 % | 16,0 % | 39,9 | 0,430 | 0,106 |
| SSH-Patator | 1.000 | 0,004 | 0,000 | ruído¹ | 99,5 % | 5,2 | 0,000 | 0,006 |

¹ TPR normal de 0,4–2 %: a taxa de vítima não é estimável nessa base. A coluna
de parada mostra que o atacante **consegue** parar a regra (82–99 % dos casos),
mas a regra já não detectava essas amostras.

**Macro nos 9 ataques com n = 1.000: 0,392 → 0,151 (−61,5 %).**
**Macro nos 11 ataques: 0,480 → 0,275 (−42,7 %).**

### 6.1 O que a tabela mostra

1. **A parada é acionável.** Em Hulk, 97,1 % das vítimas foram descartadas com
   5 máscaras; em GoldenEye, 83,9 %. O ataque consegue transformar a economia em
   evasão.
2. **Mas a parada não é a causa.** O RePO de média com **100 máscaras** sobre as
   mesmas vítimas detecta quase o mesmo tanto (`RePO 100 ≈ vítima` em todas as
   linhas). Se o problema fosse só "parar cedo", dar 100 máscaras à regra
   recuperaria a detecção — não recupera.
3. **O escore é a causa.** O RePO+ de 100 máscaras sobre as **mesmas** vítimas
   recupera a maior parte: Hulk 0,018 → 0,991; Slowhttptest 0,074 → 0,938;
   GoldenEye 0,056 → 0,719; slowloris 0,129 → 0,660. A diferença entre as duas
   colunas finais é o valor do agregado por mínimos, que a cascata preserva e a
   regra sequencial não.
4. **PortScan é a exceção instrutiva**: o RePO+ cai para 0,106 porque o TPR
   normal dele nesse ataque já é 0,11 — não há o que recuperar. A média em 0,432
   é que é anômala (o ataque baixou o escore médio mas não o suficiente para o
   agregado por grupos).

## 7. Comparação com a cascata

| | cascata cheap10 | sequencial bloco 5 |
|---|---|---|
| escore da decisão final | RePO+ (100) para quem é roteado | média (até 100) |
| máscaras/amostra (99/1, FPR 1 %) | 60,8 | 5,32 |
| TPR normal preservada | **sim** (0,25596 = 0,25596) | referência 0,2438 |
| ataque ao estágio barato | bypass 0,00 em 9/11 | — |
| ataque à regra de parada | — | **−61,5 % de TPR macro** |
| mecanismo da fragilidade | — | escore médio (não a parada) |

As duas direções respondem à mesma pergunta por caminhos opostos: a cascata
gasta mais (60,8) e mantém o agregado robusto; a sequencial gasta 5,32 e troca o
agregado pelo mais frágil. Para o artigo, isso posiciona a cascata como a
melhoria defensável e a sequencial como um limite inferior de custo com um preço
adversarial mensurável.

## 8. Ameaças à validade

- **1.000 amostras por ataque**; Heartbleed (11) e Infiltration (36) têm taxas
  instáveis.
- **O atacante para no critério pontual** (`média < τ`), enquanto a regra exige
  o limite superior do IC abaixo de `τ`. Logo, `early_stop_rate` é um **limite
  inferior**: um atacante disposto a otimizar até o IC fechar pararia ainda mais.
- **Blocos 10 e 25** foram executados; os números constam na seção 9 quando a
  rodada corrigida fechar (a primeira tentativa usou o agregado RePO+ como
  critério de sucesso, estatística diferente da média para `b > 5`, e foi
  descartada).
- **O threshold é de 10 % de FPR**, o protocolo de ataque da linha B; os números
  normais a 1 % estão no documento da cascata.
- **Sem ataque às máscaras em si**: elas são iid e derivadas de semente, não há
  escolha de máscara a corromper.
- **Uma perturbação por amostra**, não um ataque de conjunto.

## 9. Reprodutibilidade

```
# bloco 5 (headline) — 11 ataques x 1.000
MASK_SEQ_BLOCKS=5 MASK_PER_ATTACK=1000 MASK_SEQ_LOSS=ci \
MASK_ATTACK_SET=$RESULTS/mask_efficiency/attack-lists/seq-priority.txt \
bash cluster/submit.sh mask-exp sequential-attack all

# blocos 10 e 25
MASK_SEQ_BLOCKS=10,25 MASK_PER_ATTACK=1000 MASK_SEQ_LOSS=ci \
MASK_ATTACK_SET=... bash cluster/submit.sh mask-exp sequential-attack all

# consolidação
bash cluster/submit.sh mask-exp merge
```

Jobs: bloco 5 = `32174` (array 0-10); blocos 10/25 = `32212` (array 0-10).
Código: `mask_experiments/sequential_attack.py`; objetivo e escala em
`new_data_eval/attack.py` (`ci_objective`, `score_mode`); limite do IC
compartilhado em `new_data_eval/decision.py` (`mean_confidence_bounds`).

Custo da execução (bloco 5, por ataque): 55 s a 91 min; o total da onda foi
~1,5 h, dominado pelo DDoS (91 min — o único em que o atacante falhou em 46 % das
amostras e gastou o orçamento inteiro).

## 10. Checagem de números

1. **Macros**: média simples das colunas `normal` e `victim`. Nos 9 ataques com
   n = 1.000, (0,018+0,681+0,399+0,657+0,405+0,347+0,020+0,997+0,004)/9 = 0,3918
   e (0,012+0,611+0,056+0,017+0,068+0,118+0,043+0,432+0,000)/9 = 0,1508, queda
   61,5 %. Nos 11, somando Heartbleed (1,000 → 1,000) e Infiltration
   (0,750 → 0,667): 0,4798 → 0,2749, queda 42,7 %.
2. **A parada não explica a queda**: nos 11 ataques, `|victim_tpr −
   transfer_full100_tpr| ≤ 0,011` em todos, exceto Infiltration (0,083 com
   n = 36), enquanto `repo_plus_full100_tpr` chega a ser 55× maior (Hulk: 0,991
   vs 0,018).
3. **Custo normal**: 8,05 máscaras/amostra nos benignos e 13,55 nos ataques,
   contra 100 do RePO+ — a economia de ~92 % é consistente com a regra parando
   em 5–10 máscaras na maioria das amostras.
