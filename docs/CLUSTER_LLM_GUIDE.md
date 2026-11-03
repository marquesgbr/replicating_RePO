# Guia operacional do cluster Apuana para agentes LLM

Manual para uma sessão de agente **sem contexto prévio** operar o cluster Apuana
do CIn/UFPE neste projeto. Leia inteiro antes de executar qualquer comando.

> **Regra de segurança:** este guia só usa comandos de leitura para diagnóstico.
> Não cancele, não submeta, não altere e não reinicie jobs sem autorização
> explícita do usuário. O job array **18836** está em andamento e **não pode ser
> cancelado, requeueado ou sobrescrito**.

## 1. Visão geral

| Item | Valor |
|---|---|
| Cluster | Apuana, CIn/UFPE |
| Usuário | `glp3` |
| Host de login | `slurm-client1.cin.ufpe.br` (alternativa: `slurm-client2.cin.ufpe.br`) |
| IP quando a VPN está ativa | `172.21.6.31` |
| Acesso à rede | VPN IPSec do CIn **obrigatória** |
| Autenticação | chave SSH; `BatchMode=yes` funciona sem prompt |
| Gerenciador de filas | SLURM 22.05.3 |
| Login | `slurm-client1`, Xeon X5667, 8 CPUs, ~7 GB RAM, **sem AVX** |
| Nós de computação | `cluster-node1` … `cluster-node10`; GPU NVIDIA com driver `590.48.01` |

Não existe alias de host em `~/.ssh/config` na máquina local usada neste
projeto. Use sempre o FQDN:

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 glp3@slurm-client1.cin.ufpe.br
```

O caminho do repositório no cluster é o mesmo `~/Projects/replicating_RePO`,
que resolve para:

```text
/home/CIN/glp3/Projects/replicating_RePO
```

## 2. Pré-requisitos e diagnóstico rápido

### 2.1 VPN primeiro

A VPN IPSec do CIn precisa estar conectada **antes** de qualquer SSH. Teste o
DNS local:

```bash
getent hosts slurm-client1.cin.ufpe.br
```

Saída esperada com VPN ativa:

```text
172.21.6.31     slurm-client1.cin.ufpe.br
```

Se o comando não retornar nada ou retornar `Could not resolve hostname`, a VPN
caiu. Reconecte o cliente IPSec do CIn e repita. Sem VPN, nenhum comando deste
guia funciona.

### 2.2 SSH

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 glp3@slurm-client1.cin.ufpe.br \
  'hostname; whoami; date; date -u; /usr/bin/python3.10 --version'
```

Saída esperada (exemplo):

```text
slurm-client1
glp3
Tue Nov  3 12:22:22 PM -03 2026
Tue Nov  3 03:22:22 PM UTC 2026
Python 3.10.12
```

### 2.3 Checklist mínimo antes de operar

| Verificação | Comando | Esperado |
|---|---|---|
| VPN/DNS | `getent hosts slurm-client1.cin.ufpe.br` | `172.21.6.31` |
| SSH sem prompt | `ssh -o BatchMode=yes ... 'hostname'` | `slurm-client1` |
| Repositório | `git -C ~/Projects/replicating_RePO status --short` | limpo ou só untracked conhecido |
| Venv | `~/replicating-repo-venv/bin/python --version` | `Python 3.10.12` |
| Fila | `squeue -u glp3` | estado atual dos jobs |
| Nós | `sinfo -o "%20P %10a %10l %10D %10T %N"` | partições e estados |

## 3. Mapa de caminhos no cluster

`$HOME` é `/home/CIN/glp3`. `$SCRATCH` **não vem definido** em um shell SSH
novo; o `cluster/config.env` do cluster define `SCRATCH=$HOME`.

| O que | Caminho absoluto | Observações |
|---|---|---|
| Repositório | `/home/CIN/glp3/Projects/replicating_RePO` | branch `feat/apuana-cluster` |
| Config local do cluster | `.../replicating_RePO/cluster/config.env` | gitignored, só existe no cluster |
| Venv | `/home/CIN/glp3/replicating-repo-venv` | `bin/python -> /usr/bin/python3.10` |
| Dados extraídos | `/home/CIN/glp3/replicating-repo-data` | ~2,6 GB |
| Symlinks de dados | `.../replicating_RePO/data/flow_based/*.csv` e `.../data/packet_based/{monday..friday}` | apontam para `replicating-repo-data` |
| Metadados de normalização | `.../data/flow_based/x_train_meta/` e `.../data/packet_based/x_train_meta/` | `train_min.npy`, `train_max.npy` |
| Resultados | `/home/CIN/glp3/replicating-repo-results` | JSON/NPZ por execução |
| Logs SLURM | `.../replicating-repo-results/logs/` | `*.out`, `*.err`, `*.log` |
| Checkpoints | `.../replicating-repo-results/{flow,packet}/adversarial/checkpoints/<Ataque>.npz` | retomada |
| Listas de ataques | `.../replicating-repo-results/attack-lists/` | uma linha por ataque |
| Manifestos | `.../replicating-repo-results/manifests/` | ambiente, git, fingerprints |
| Modelos dos autores | `.../replicating_RePO/models/flw_model` e `.../models/pkt_model` | **não sobrescrever** |
| Modelos retreinados (opcional) | `.../models/retrained/{flw_model,pkt_model}` | só com `MODEL_SOURCE=retrained` |
| Resumo agregado | `.../replicating_RePO/results/cluster/{summary.csv,summary.md,summary.json}` | gerado por `submit.sh export` |

Layout de resultados:

```text
~/replicating-repo-results/
├── flow/
│   ├── normal/{repo.json,repo_plus.json,scores_*.npz}
│   └── adversarial/{<Ataque>.json,checkpoints/<Ataque>.npz,scores_<Ataque>.npz}
├── packet/
│   ├── normal/{repo.json,repo_plus.json,scores_*.npz}
│   └── adversarial/{<Ataque>.json,checkpoints/<Ataque>.npz,scores_<Ataque>.npz}
├── logs/
├── manifests/
├── attack-lists/
└── smoke/
```

## 4. Ambiente de software

| Componente | Versão | Onde |
|---|---|---|
| Python | 3.10.12 | `/usr/bin/python3.10` em todos os nós |
| TensorFlow | 2.21.0 | venv `~/replicating-repo-venv` |
| Keras | 3.12.4 | venv |
| NumPy | 2.2.6 | venv |
| Pandas | 2.2.3 | venv |
| CUDA/cuDNN | wheels `nvidia-*-cu12` via pip | dentro do venv |
| Driver NVIDIA | 590.48.01 | nós de computação |
| GPU observada | A100 80GB PCIe e RTX 3090 24GB | depende do nó |
| SLURM | 22.05.3 | login e nós |

### 4.1 Por que o venv não pode vir do miniforge

`/opt/miniforge3` existe **apenas nos nós de login** e tem Python 3.12.10. Um
venv criado com ele grava um `bin/python` que aponta para
`/opt/miniforge3/bin/python3.12`; nos nós de computação esse caminho não existe,
e o job morre com `python: No such file or directory` ou
`bad interpreter`.

O venv correto é criado com o Python do sistema:

```bash
/usr/bin/python3.10 --version        # 3.10.12
readlink -f ~/replicating-repo-venv/bin/python   # /usr/bin/python3.10
```

Se o venv estiver quebrado, recrie **no login**:

```bash
cd ~/Projects/replicating_RePO
bash cluster/setup_env.sh --force-venv
```

Não defina `PYTHON_MODULE` no `cluster/config.env`.

### 4.2 Por que Keras está pinado em 3.12.4

Keras 3.13+ exige Python ≥ 3.11. O teto do Apuana é Python 3.10.12, então
`cluster/requirements-cluster.txt` fixa:

```text
tensorflow[and-cuda]==2.21.0
keras==3.12.4
```

Não rode `pip install -U keras` nem `pip install -U tensorflow` dentro do venv.
Isso quebra o ambiente dos jobs e pode invalidar comparações.

## 5. Regra de ouro: nunca rode TensorFlow no nó de login

O login é um **Xeon X5667 sem AVX** (confirmado: `grep avx /proc/cpuinfo` não
retorna nada). O TensorFlow 2.21 compilado usa instruções AVX e aborta:

```text
Illegal instruction (core dumped)
```

Isso **não** é defeito da instalação. Toda validação de TensorFlow, Keras, CUDA
ou GPU deve ocorrer em nó de computação, via `srun`/`sbatch`.

| Pode rodar no login | Não pode rodar no login |
|---|---|
| `git`, `ls`, `cat`, `grep`, `find` | `import tensorflow` |
| `bash -n script.sh` | `python cluster/repo_runner.py ...` |
| `python -m py_compile arquivo.py` | smoke test |
| `squeue`, `sinfo`, `scontrol` | treino, avaliação normal/adversarial |
| `bash cluster/submit.sh ...` (só submete) | `nvidia-smi` com GPU |

Validação interativa em nó de computação:

```bash
cd ~/Projects/replicating_RePO
srun --partition=short-simple \
  --exclude=cluster-node4,cluster-node10 \
  --gres=gpu:1 -c 8 --mem=16G --time=00:10:00 --pty bash

# já dentro do nó:
source ~/replicating-repo-venv/bin/activate
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
```

Nunca use `--nodelist=cluster-node5,cluster-node6`: isso pede **um nó por host**
(ver seção 6.3). Para fixar um único nó, use um só host, por exemplo
`--nodelist=cluster-node6`.

## 6. SLURM: partições, limites e nós

### 6.1 Partições

| Partição | Interativa | Recursos | Tempo máximo |
|---|---|---|---|
| `install` | sim | meio nó, até 2 GPUs | 30 min |
| `debug` | sim | nó completo | 30 min |
| `short-simple` | não | meio nó (até 2 GPUs) | 2 dias |
| `short-complex` | não | nó completo (até 4 GPUs) | 2 dias |
| `long-simple` | não | meio nó (até 2 GPUs) | 7 dias |
| `long-complex` | não | nó completo (até 4 GPUs) | 7 dias |

O `submit.sh` usa `PARTITION=short-simple` por padrão.

### 6.2 Limites observados (nov/2026)

| Limite | Valor | Consequência prática |
|---|---|---|
| GPUs simultâneas por usuário | **2** | `CONCURRENCY=2`, `GRES=gpu:1` |
| Memória simultânea por usuário | **~32 GB no total** | `MEM=16G` por job |
| CPUs por job | 8 no default | `CPUS_PER_TASK=8` |
| QoS das partições | `QOS=simple` | exceder memória/GPU gera erro de QoS |
| Array | `--array=0-N%2` | no máximo 2 tarefas rodando |

**Armadilha:** o `cluster/config.env.example` sugere `MEM=32G`. Dois jobs
concorrentes com 32G pedem 64G e batem em `QOSMaxMemoryPerUser`. O
`cluster/config.env` real do cluster usa `MEM=16G`; mantenha 16G.

### 6.3 Nós com defeito e a armadilha do `--nodelist`

Estado observado em 2026-11-03:

| Nó | Situação observada | Ação |
|---|---|---|
| `cluster-node4` | comunicação instável / falha de launch | excluir |
| `cluster-node10` | `DOWN+NOT_RESPONDING`, reason `hardware` | excluir |
| `cluster-node9` | já esteve ocioso mas com `srun` falhando | testar antes; se falhar, excluir |
| `cluster-node3` | funciona; RTX 3090 24GB | ok |
| `cluster-node5`, `cluster-node6`, `cluster-node7` | funcionam | ok |

Configuração recomendada:

```bash
export EXCLUDE="cluster-node4,cluster-node10"
```

Por que **não** usar `--nodelist` com vários hosts:

- `--nodelist=a,b,c` significa “o job deve conter esses hosts”; o Slurm trata
  cada host listado como um nó solicitado.
- Com `-N 1` explícito o comportamento pode variar, mas no uso observado o job
  passou a pedir **um nó por host**, multiplicando CPUs, memória e GPUs.
- O resultado é fila eterna (`PD`) ou erro de QoS, não “escolha um dos nós”.

`--exclude` apenas remove nós ruins da seleção e **não altera a contagem de
nós**. Prefira:

```bash
EXCLUDE="cluster-node4,cluster-node10" bash cluster/submit.sh adversarial packet all
```

Se precisar de um nó específico, use **um único** host e apenas depois de
testá-lo:

```bash
NODELIST=cluster-node6 bash cluster/submit.sh adversarial flow all
```

### 6.4 GPU: A100 ou RTX 3090

O cluster tem modelos diferentes de GPU. Pelos manifestos e logs:

- `cluster-node9`: **NVIDIA A100 80GB PCIe**;
- `cluster-node3`: **NVIDIA GeForce RTX 3090 24GB** (para onde foram a maioria
  dos jobs recentes).

O `--gres=gpu:1` não escolhe modelo. Se o tempo importar, verifique em qual nó o
job caiu (`scontrol show job <id> | grep NodeList` e o cabeçalho do `.out`). Não
fixe `--nodelist` em vários hosts para “pegar A100”.

## 7. Submissão de jobs

### 7.1 Carregar a configuração

```bash
cd ~/Projects/replicating_RePO
set -a
source cluster/config.env
set +a
```

`cluster/config.env` é **gitignored** e vive apenas no cluster. O `submit.sh`
também o carrega automaticamente, mas o `source` é necessário para comandos
manuais e para exportar variáveis ao shell.

Variáveis relevantes:

| Variável | Default no cluster | Uso |
|---|---|---|
| `PARTITION` | `short-simple` | partição do job |
| `CPUS_PER_TASK` | `8` | CPUs por tarefa |
| `MEM` | `16G` | memória por job; não aumentar sem necessidade |
| `GRES` | `gpu:1` | uma GPU por job |
| `CONCURRENCY` | `2` | tarefas simultâneas do array |
| `EXCLUDE` | `cluster-node4,cluster-node10` | nós ruins |
| `SEED` | `2019` | reprodutibilidade e retomada |
| `MODEL_SOURCE` | `pretrained` | modelos dos autores; `retrained` para os do cluster |
| `MODEL_DIR` | vazio | caminho explícito de modelo |
| `FORCE` | `0` | `1` descarta resultado/checkpoint/scores |
| `MAX_SAMPLES` | vazio | limita amostras/pacotes do ataque |
| `PACKET_GPU_LOOP` | `0` | `1` ativa loop compilado de pacotes |
| `WALLTIME_TRAIN` | `1-00:00:00` | treino |
| `WALLTIME_NORMAL` | `06:00:00` | avaliação normal |
| `WALLTIME_ADV` | `1-23:00:00` | adversarial (47 h) |
| `CHECKPOINT_EVERY` | `50` | intervalo de checkpoint |
| `LOG_EVERY` | `50` | intervalo de log de progresso |
| `NODELIST` | vazio | **um** host, se realmente necessário |

### 7.2 Comandos

```bash
# avaliação normal (RePO e RePO+ no mesmo job)
bash cluster/submit.sh normal flow
bash cluster/submit.sh normal packet

# um ataque específico (nome exato)
bash cluster/submit.sh adversarial flow "DoS Slowhttptest"
bash cluster/submit.sh adversarial packet "FTP-Patator"

# todos os 11 ataques, um por tarefa do array
bash cluster/submit.sh adversarial flow all
PACKET_GPU_LOOP=1 bash cluster/submit.sh adversarial packet all

# piloto curto
MAX_SAMPLES=200 PACKET_GPU_LOOP=1 \
  bash cluster/submit.sh adversarial packet "FTP-Patator"

# modelos retreinados
MODEL_SOURCE=retrained bash cluster/submit.sh normal flow

# smoke não interativo (se não couber no srun)
SMOKE=1 bash cluster/submit.sh smoke flow

# agregar resultados
bash cluster/submit.sh export
```

Nomes exatos dos ataques:

| Modelo | Ataques |
|---|---|
| Fluxo (11) | `Bot`, `DDoS`, `DoS-GoldenEye`, `DoS-Hulk`, `DoS-Slowhttptest`, `DoS-slowloris`, `FTP-Patator`, `Heartbleed`, `Infiltration`, `PortScan`, `SSH-Patator` |
| Pacotes (11) | `FTP-Patator`, `SSH-Patator`, `Slowloris`, `Slowhttptest`, `Hulk`, `GoldenEye`, `Heartbleed`, `Infiltration`, `Botnet`, `PortScan`, `DDoS` |

`Web-Attack` é excluído dos pacotes, como no artigo.

### 7.3 Como o array é montado

- O `submit.sh` cria uma lista em
  `~/replicating-repo-results/attack-lists/<modelo>-<timestamp>.txt`.
- Sobe **um job array** com `--array=0-(N-1)%CONCURRENCY`; cada índice lê uma
  linha da lista e executa um ataque.
- `all` = 11 tarefas; ataque único = 1 tarefa (`0-0%2`).
- Logs únicos: `%x-%A_%a.out` / `.err`, por exemplo
  `repo-adv-packet-18836_0.out`.
- `%A` é o ID do array (18836); `%a` é o índice da tarefa (0, 1, …).
- Atenção: o `SLURM_JOB_ID` **dentro** da tarefa é outro número (por exemplo,
  tarefa 0 = 18837, tarefa 1 = 18838). Para cancelar o array, use o ID do
  array (`scancel 18836`); para uma tarefa, use o ID dela. **Não faça isso no
  18836 sem autorização.**

### 7.4 A armadilha do `--export` e do `SCORE_MODE`

O `--export` do SLURM separa variáveis por **vírgula**. Se o valor de uma
variável contiver vírgula, ele a interpreta como outra variável.

Por isso o `submit.sh` monta:

```text
SCORE_MODE=repo+repo_plus
```

e o `job.sh` converte `+` de volta para `,`:

```bash
MODES="${SCORE_MODE:-repo,repo_plus}"
MODES="${MODES//+/,}"
```

Nunca passe `SCORE_MODE=repo,repo_plus` diretamente: o SLURM criaria uma
variável `repo_plus` separada e o job falharia. O mesmo vale para caminhos com
vírgula: evite.

## 8. Checkpoint e retomada

| Item | Valor |
|---|---|
| Intervalo | a cada **50** amostras (fluxo) ou **50** pacotes |
| Arquivo | `.../adversarial/checkpoints/<Ataque>.npz` |
| Conteúdo | `next_index`, `normal_scores`, estado da otimização, `config_hash` |
| Retomada | reenviar o **mesmo** comando |
| Resultado concluído | nova submissão é no-op (idempotência) |
| `config_hash` | precisa bater exatamente |

O `config_hash` cobre modelo, fingerprint dos dados, seed, thresholds,
`max_samples`, `day`/`label` dos pacotes e `packet_gpu_loop`. Se qualquer um
mudar, o runner recusa:

```text
checkpoint encontrado com configuração diferente; use --force
```

Opções:

1. **Correto:** repetir os mesmos parâmetros e continuar do checkpoint.
2. **Destrutivo:** `FORCE=1` apaga o JSON de resultado, o checkpoint e o
   `scores_*.npz` daquele ataque e começa do zero.

```bash
# retomar sem perder progresso
bash cluster/submit.sh adversarial packet "FTP-Patator"

# ATENÇÃO: descarta resultado, checkpoint e scores do ataque
FORCE=1 bash cluster/submit.sh adversarial packet "FTP-Patator"
```

**Nunca use `FORCE=1` por reflexo.** Em particular, não use no array 18836 nem
em ataques já concluídos. Se o hash mudou por causa de `PACKET_GPU_LOOP`, a
decisão de descartar o checkpoint é do usuário, não do agente.

O `--requeue` está ativo nos jobs: falha de nó pode reiniciar a tarefa, que
retoma do último checkpoint.

## 9. Loop compilado de pacotes (`PACKET_GPU_LOOP=1`)

O ataque de pacotes original faz 300 iterações de otimização por pacote em
Python, sincronizando CPU↔GPU a cada passo. A branch `main` adotou um
`tf.while_loop` compilado (`compiled_delay_loop` / `compiled_inject_loop`); o
runner ativa isso com `--packet-gpu-loop`, via `PACKET_GPU_LOOP=1`.

Validação A/B em 200 pacotes do FTP-Patator, mesma GPU e mesma seed:

| Caso | Tempo do ataque | `victim_tpr` | `fooled` |
|---|---|---|---|
| sem loop compilado | 245,1 s | 0,6350 | 3 |
| `PACKET_GPU_LOOP=1` | **94,3 s** | 0,6350 | 3 |

Ganho de **2,6×** com resultado idêntico. Sem a flag, o ataque de pacotes fica
aproximadamente 2,6× mais lento.

A flag entra no `config_hash`. Consequências:

- checkpoint antigo sem a flag **não** é reaproveitado por engano;
- para mudar de modo, o runner exige `FORCE=1` (decisão consciente);
- a rodada adversarial de pacotes em andamento (array **18836**) usa
  `PACKET_GPU_LOOP=1`.

```bash
PACKET_GPU_LOOP=1 bash cluster/submit.sh adversarial packet all
```

## 10. Vigia remoto e ponte local

### 10.1 Por que destacar no cluster

Se a máquina local suspender, cair a rede ou o SSH fechar:

- os **jobs do SLURM continuam** rodando no cluster;
- um vigia iniciado com `setsid nohup ... < /dev/null` **continua** rodando;
- a **ponte local** (loop de `ssh` na sua máquina) morre e precisa ser
  reiniciada.

Nunca dependa de um `while` local para “manter o job vivo”. O job não depende
da sua sessão SSH.

### 10.2 Script de vigia que não dá falso positivo

O script abaixo espera o job sair da fila e só declara `CONCLUIDO` se houver o
número esperado de JSONs. Se o array for cancelado ou falhar, grava
`INCOMPLETO`.

```bash
cat > ~/watch_array.sh <<'EOF'
#!/usr/bin/env bash
# Uso: watch_array.sh <array_job_id> <jsons_esperados> <dir_resultados>
set -uo pipefail

JOB_ID="${1:?informe o ID do array}"
EXPECTED="${2:?informe o numero esperado de JSONs}"
DIR="${3:?informe o diretorio de resultados}"
OUT="$HOME/watcher-${JOB_ID}.done"

while squeue -h -u "$USER" -j "$JOB_ID" 2>/dev/null | grep -q .; do
  sleep 120
done

count=$(find "$DIR" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)

if (( count >= EXPECTED )); then
  echo "CONCLUIDO job=$JOB_ID em $(date -u) ($count/$EXPECTED JSONs)" > "$OUT"
else
  echo "INCOMPLETO job=$JOB_ID em $(date -u) ($count/$EXPECTED JSONs)" > "$OUT"
  echo "Verifique .err, checkpoints e o motivo em squeue/scontrol." >> "$OUT"
fi

echo "--- resultados ---" >> "$OUT"
ls -1 "$DIR"/*.json 2>/dev/null | xargs -r -n1 basename | sort >> "$OUT"
EOF

chmod +x ~/watch_array.sh

# inicia destacado; sobrevive ao fechamento do SSH
setsid nohup ~/watch_array.sh 18836 11 \
  "$HOME/replicating-repo-results/packet/adversarial" \
  </dev/null >/dev/null 2>&1 &
```

O vigia existente no cluster segue o mesmo padrão:

```text
~/watch_packet.sh 18836 11   ->   ~/watcher-packet.done
```

Leitura:

```bash
cat ~/watcher-packet.done
```

**Limitação:** se o diretório já contém JSONs de rodadas anteriores, a contagem
pode chegar ao total antes da rodada atual terminar. Para uma rodada nova,
arquive/limpe apenas com autorização; nunca apague resultados de jobs em
andamento. Para o array atual, o diretório de pacotes ainda não tinha JSONs
concluídos no momento da escrita.

### 10.3 Ponte local

Na máquina local, consulte o arquivo a cada 2–3 minutos:

```bash
while true; do
  msg=$(ssh -o BatchMode=yes -o ConnectTimeout=15 \
    glp3@slurm-client1.cin.ufpe.br \
    'cat ~/watcher-packet.done 2>/dev/null' || true)
  if [[ -n "$msg" ]]; then
    printf '%s\n' "$msg"
    break
  fi
  sleep 120
done
```

Se a máquina local suspender, esse loop para. Ao voltar, rode de novo: o
arquivo remoto já estará lá.

## 11. Fuso horário: por que os logs parecem 3 h atrasados

| Contexto | Fuso | Exemplo |
|---|---|---|
| Nós de computação e login (`date`) | UTC−3 (horário local) | `Tue Nov  3 12:22:22 PM -03 2026` |
| `date -u` no login | UTC | `Tue Nov  3 03:22:22 PM UTC 2026` |
| Linhas dentro do `.log` do runner | UTC−3 (horário local do nó) | `2026-11-03 11:56:33` |
| Nome do `.log` (`utc_stamp()`) | UTC | `20261103-145628_job18837_...` |
| `watcher-*.done` (`date -u`) | UTC | `Tue Nov  3 05:50:25 AM UTC 2026` |

Portanto, um log com `11:56` e um `date -u` marcando `15:00` **não estão
atrasados**: são o mesmo instante em fusos diferentes. Ao correlacionar horários,
escolha uma referência só (prefira UTC).

## 12. Diagnóstico de problemas

| Sintoma | Causa provável | Correção |
|---|---|---|
| `Illegal instruction (core dumped)` ao importar TF no login | CPU do login sem AVX | normal; valide TF apenas em `srun`/`sbatch` |
| `Could not resolve hostname slurm-client1...` | VPN IPSec caiu | reconectar VPN; testar `getent hosts` |
| `QOSMaxMemoryPerUser` | soma de memória dos jobs passou de ~32 GB | usar `MEM=16G` e `CONCURRENCY=2`; esperar job terminar |
| `QOSMaxGRESPerUser` | mais de 2 GPUs simultâneas | `GRES=gpu:1`, `CONCURRENCY=2` |
| `JobArrayTaskLimit` | throttle normal do array (`%2`) | não é erro; tarefas 2–10 aguardam |
| `python: No such file or directory` no nó | venv criado com miniforge (só no login) | recriar com `/usr/bin/python3.10` (`setup_env.sh --force-venv`) |
| `No module named tensorflow` | venv não ativado | `source ~/replicating-repo-venv/bin/activate` |
| `checkpoint ... configuração diferente` | `config_hash` mudou (seed, modelo, `max_samples`, `packet_gpu_loop`, threshold) | repetir os mesmos parâmetros; `FORCE=1` só com intenção |
| `--nodelist` com vários hosts pede vários nós | cada host vira um nó solicitado | usar `EXCLUDE`; para fixar, um único host |
| Último lote de pacotes “descartado” | comportamento do notebook antigo | o runner atual pontua todas as janelas; não há ação |
| Rótulos de pacote vêm como `float64` | dtype do dataset original | o runner converte para `float32`; não há ação |
| Job preso em `PD` | recursos/QoS ou nó indisponível | `squeue -j <id> -o %r`; ajustar `MEM`/`GRES`/`EXCLUDE` |
| Job morre no launch em node4/node10 | nó com defeito | `EXCLUDE="cluster-node4,cluster-node10"` |
| `srun` falha no node9 mesmo ocioso | problema de launch já observado | testar; se persistir, excluir node9 |
| `sacct: ... Connection refused` | `slurmdbd` fora do ar | usar `squeue`, `scontrol` e logs |
| GPU não aparece em `tf.config.list_physical_devices` | job sem `--gres=gpu:1` ou rodando no login | submeter com GPU e validar no nó |
| `FileNotFoundError` de CSV/`.npy` | dados não extraídos ou symlink quebrado | `bash cluster/setup_env.sh --force-data` |
| Pouco espaço | dados + venv + resultados | `df -h ~`; arquivar resultados com autorização |
| Logs “3 h atrasados” | fuso UTC−3 vs UTC | ver seção 11 |

## 13. Comandos de monitoramento prontos

### 13.1 Fila e estado

```bash
ssh -o BatchMode=yes -o ConnectTimeout=15 glp3@slurm-client1.cin.ufpe.br \
  'squeue -u glp3 -o "%.12i %.28j %.8T %.10M %R"'
```

Exemplo de leitura:

```text
JOBID        NAME                         STATE      TIME       NODELIST(REASON)
18836_[2-10% repo-adv-packet              PENDING    0:00       (JobArrayTaskLimit)
18836_0      repo-adv-packet              RUNNING    26:06      cluster-node3
18836_1      repo-adv-packet              RUNNING    26:06      cluster-node3
```

```bash
# detalhes de um array/tarefa
ssh ... 'scontrol show job 18836 | grep -E "JobState|RunTime|NodeList|ExcNodeList|NumNodes|TRES|Reason"'

# partições e nós
ssh ... 'sinfo -o "%20P %10a %10l %10D %10T %N"'

# só jobs rodando
ssh ... 'squeue -u glp3 -h -t RUNNING | wc -l'
```

### 13.2 Logs

```bash
# últimas linhas do array de pacotes
ssh ... 'tail -n 60 ~/replicating-repo-results/logs/repo-adv-packet-18836_0.out'

# acompanhar ao vivo
ssh ... 'tail -f ~/replicating-repo-results/logs/repo-adv-packet-18836_0.out'

# erros
ssh ... 'tail -n 40 ~/replicating-repo-results/logs/repo-adv-packet-18836_0.err'

# progresso por checkpoint
ssh ... 'grep -c "checkpoint salvo" ~/replicating-repo-results/logs/repo-adv-packet-18836_0.out'
```

### 13.3 Resultados e checkpoints

```bash
# JSONs concluídos do adversarial de pacotes
ssh ... 'find ~/replicating-repo-results/packet/adversarial -maxdepth 1 -name "*.json" | wc -l'

# quais ataques já terminaram
ssh ... 'ls -1 ~/replicating-repo-results/packet/adversarial/*.json 2>/dev/null'

# checkpoints e progresso
ssh ... 'ls -lh ~/replicating-repo-results/packet/adversarial/checkpoints/'

# vigia
ssh ... 'cat ~/watcher-packet.done 2>/dev/null'

# normal
ssh ... 'ls -1 ~/replicating-repo-results/flow/normal ~/replicating-repo-results/packet/normal'
```

### 13.4 `sacct`

O comando pretendido é:

```bash
sacct -u glp3 --starttime=today \
  --format=JobID%20,JobName%30,State,Elapsed,NodeList
```

Mas, no momento da escrita, o `slurmdbd` estava fora:

```text
sacct: error: ... Connection refused
```

Se falhar, use `squeue`, `scontrol show job` e os arquivos `.out`/`.err`.

## 14. Fluxo recomendado para um agente novo

1. **VPN:** `getent hosts slurm-client1.cin.ufpe.br` → `172.21.6.31`.
2. **SSH:** `ssh -o BatchMode=yes -o ConnectTimeout=15 glp3@... 'hostname'`.
3. **Repositório:** entrar em `~/Projects/replicating_RePO`, rodar
   `git status --short` e `git log --oneline -3`.
   Se for atualizar: `git pull --ff-only`. **Nunca** `git reset --hard` nem
   `git clean -fd` no cluster: há arquivos untracked (dados, pilotos) e jobs em
   andamento. Se houver dúvida, pare e pergunte.
4. **Configuração:** `set -a; source cluster/config.env; set +a` e conferir
   `MEM=16G`, `EXCLUDE=cluster-node4,cluster-node10`, `CONCURRENCY=2`.
5. **Fila:** `squeue -u glp3 -o "%.12i %.28j %.8T %.10M %R"`; identificar jobs
   ativos e **não** duplicar submissões.
6. **Ambiente:** validar TF/GPU em nó de computação com `srun` (seção 5) ou
   rodar o smoke. **Nunca** no login.
7. **Submeter** só depois de saber o que já está rodando e qual é o objetivo.
   Preferir um ataque/piloto antes de `all`.
8. **Monitorar** por `squeue`, logs e JSONs; usar o vigia remoto para jobs
   longos.
9. **Nunca** apagar `~/replicating-repo-results`, checkpoints ou modelos sem
   autorização explícita. **Nunca** cancelar o array 18836.
10. Ao terminar, agregar com `bash cluster/submit.sh export` e reportar os
    números, sem alterar resultados brutos.

## 15. Fatos do projeto úteis

| Fato | Detalhe |
|---|---|
| Artigo | RePO (avaliação de NIDS em cenário adversarial) |
| Ataques por modelo | 11 |
| Fluxo | primeiras **5.000** amostras de cada ataque |
| Pacotes | primeiros **25.000** pacotes de cada ataque |
| FPR normal | 0,01 |
| FPR adversarial | 0,10 |
| Modelos | pré-treinados dos autores; **não retreinamos** por padrão |
| Normal de fluxo | RePO 0,2443; RePO+ 0,2525 |
| Normal de pacotes | RePO 0,3185; RePO+ 0,3472 |
| FPR normal medido | 0,0100 |
| Adversarial de fluxo | concluído; TPR vítima ponderado 0,4663 (artigo 0,4702); 8/11 acima do FPR 0,10 |
| Adversarial de pacotes | array 18836 em andamento; 11 ataques × 25.000 pacotes, `PACKET_GPU_LOOP=1` |

Thresholds de referência (`cluster/metrics.py`):

| Modelo | Threshold normal/FPR 0,01 | Threshold adversarial |
|---|---|---|
| Fluxo | calculado no conjunto benigno | `0,016235964372754097 × 0,8` |
| Pacotes | calculado no conjunto benigno | `0,08106673508882523` |

## 16. Estado no momento da escrita e o que não foi verificado

**Data da verificação:** 2026-11-03, aproximadamente 12:25 (UTC−3) /
15:25 UTC.

Verificado com comandos de leitura:

- VPN ativa; `getent hosts` retornou `172.21.6.31`;
- SSH com `BatchMode=yes` funcionando;
- caminhos do repositório, venv, dados, resultados, logs, checkpoints e
  `cluster/config.env` existem;
- Python 3.10.12, TensorFlow 2.21.0, Keras 3.12.4, NumPy 2.2.6, Pandas 2.2.3;
- driver 590.48.01 e modelo de GPU confirmados pelos manifestos dos jobs;
- `sinfo`, `squeue`, `scontrol show job 18836` e logs do array;
- array 18836 com tarefas 0 e 1 rodando (`FTP-Patator`, `SSH-Patator`) e
  tarefas 2–10 pendentes por `JobArrayTaskLimit`.

Não verificado / limitações:

- `nvidia-smi` ao vivo em nó de computação: não foi executado para não
  submeter job; a informação veio dos manifestos e logs dos jobs já rodados;
- `sacct`/`sacctmgr`: `slurmdbd` indisponível (`Connection refused`), então os
  limites de QoS foram registrados a partir da observação prática do projeto;
- comportamento atual de `srun` no `cluster-node9`: o nó estava `MIXED`, mas já
  apresentou falhas de launch; teste antes de usá-lo;
- o `PACKET_GPU_LOOP=1` do array 18836 foi registrado conforme o histórico do
  projeto; o `config_hash` do checkpoint não é legível como texto.

Se a VPN estiver fora no futuro, este guia continua válido para leitura e
planejamento, mas **nenhum caminho ou estado do cluster pode ser confirmado**
sem reconectar a VPN.
