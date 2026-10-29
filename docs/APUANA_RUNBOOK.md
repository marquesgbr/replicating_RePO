# Passo a passo — rodar a replicação no cluster Apuana

Este é o roteiro linear para executar a replicação no Apuana. O runbook de
referência, com detalhes de cada decisão e das partições, está em
[`../cluster/README.md`](../cluster/README.md).

> **Resumo do fluxo:** setup no login → smoke na partição `debug` → avaliação
> normal → piloto adversarial de fluxo → piloto adversarial de pacotes →
> rodada completa → agregação e coleta.

## 0. Antes de começar

Confirme que você tem:

- VPN do CIn ativa;
- acesso SSH a `slurm-client1.cin.ufpe.br` (ou `slurm-client2`);
- chave SSH cadastrada no GitHub para clonar o repositório;
- acesso ao cluster e espaço no `$SCRATCH` (ou `$HOME`).

Não execute treino, avaliação ou notebooks nos nós de login. Use o login
apenas para setup, submissão, agregação e transferência de arquivos.

## 1. Entrar no cluster e conferir o armazenamento

```bash
ssh <login>@slurm-client1.cin.ufpe.br

echo "HOME=$HOME"
echo "SCRATCH=${SCRATCH:-<VAZIO>}"
echo "PROJECTS=${PROJECTS:-<VAZIO>}"
echo "ARCHIVE=${ARCHIVE:-<VAZIO>}"
df -h "$HOME" ${SCRATCH:+"$SCRATCH"}
```

Se `$SCRATCH` estiver vazio, use `$HOME` como base:

```bash
export SCRATCH="$HOME"
```

O dataset extraído ocupa cerca de **13 GB** e o venv com CUDA mais alguns GB.
O padrão do tooling é:

```text
$SCRATCH/replicating-repo-venv      # ambiente Python
$SCRATCH/replicating-repo-data      # dados extraídos
$SCRATCH/replicating-repo-results   # resultados e logs
```

## 2. Clonar o repositório e entrar na branch

```bash
mkdir -p "$HOME/Projects"
cd "$HOME/Projects"

git clone git@github.com:marquesgbr/replicating_RePO.git
cd replicating_RePO
git switch feat/apuana-cluster
git log --oneline -1
```

Se o repositório já existir:

```bash
cd "$HOME/Projects/replicating_RePO"
git fetch origin
git switch feat/apuana-cluster
git pull --ff-only
```

### 2.1 Configuração persistente (recomendado)

Cada novo shell SSH pode não exportar `$SCRATCH`. Para não criar o venv em um
lugar e procurá-lo em outro, crie `cluster/config.env`:

```bash
cd "$HOME/Projects/replicating_RePO"
cp cluster/config.env.example cluster/config.env
```

Edite `cluster/config.env` se necessário (principalmente `SCRATCH` e
`PYTHON_MODULE`) e carregue-o em todo shell:

```bash
set -a
source "$HOME/Projects/replicating_RePO/cluster/config.env"
set +a
```

Opcionalmente, adicione ao `~/.bashrc`:

```bash
echo 'source "$HOME/Projects/replicating_RePO/cluster/config.env"' >> ~/.bashrc
```

O arquivo `cluster/config.env` é ignorado pelo Git e não deve ser commitado.

## 3. Descobrir o módulo Python

```bash
module avail 2>&1 | grep -i python
```

Se aparecer, por exemplo, `Python/3.10`, fixe o nome exato:

```bash
export PYTHON_MODULE="Python/3.10"
```

Se não aparecer nada, o setup tenta descobrir automaticamente. Nesse caso,
confira também se `python3 --version` é pelo menos 3.10.

## 4. Preparar o ambiente e os dados (uma vez, no login)

```bash
cd "$HOME/Projects/replicating_RePO"
export SCRATCH="${SCRATCH:-$HOME}"

bash cluster/setup_env.sh
```

O setup faz, em ordem:

1. cria `$SCRATCH/replicating-repo-venv`;
2. instala TensorFlow 2.21 + Keras 3.15.1 + CUDA via pip;
3. extrai os CSVs de fluxo em `$SCRATCH/replicating-repo-data/flow_based`;
4. extrai os `.npy` de pacotes em `$SCRATCH/replicating-repo-data/packet_based`;
5. cria symlinks em `data/` apontando para os dados extraídos;
6. valida modelos, dados e import do TensorFlow;
7. grava o `pip freeze` em
   `$SCRATCH/replicating-repo-results/env-lock.txt`.

Isso pode demorar dependendo da rede e do disco. Para repetir apenas a extração dos dados:

```bash
bash cluster/setup_env.sh --force-data
```

Para recriar o venv:

```bash
bash cluster/setup_env.sh --force-venv
```

Para refazer tudo:

```bash
bash cluster/setup_env.sh --force
```

Para pular uma etapa já feita:

```bash
bash cluster/setup_env.sh --skip-venv
bash cluster/setup_env.sh --skip-data
```

## 5. Verificações rápidas no login

Ative o ambiente:

```bash
source "$SCRATCH/replicating-repo-venv/bin/activate"
```

Confira TensorFlow e dados:

```bash
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices('GPU'))"
ls -lh data/flow_based/Tuesday-WH-generate-labeled.csv
ls data/packet_based/tuesday | head
ls -lh models/flw_model/saved_model.pb models/pkt_model/saved_model.pb
du -sh "$SCRATCH/replicating-repo-data"
```

No nó de login, a lista de GPUs pode vir vazia; isso é esperado. A GPU só é
validada no smoke, dentro de um nó de compute.

## 6. Smoke test (obrigatório)

A partição `debug` é interativa e tem limite de 30 minutos:

```bash
cd "$HOME/Projects/replicating_RePO"

srun --partition=debug --gres=gpu:1 -c 8 --mem=32G --time=00:30:00 --pty bash
```

Já dentro do nó:

```bash
source "$SCRATCH/replicating-repo-venv/bin/activate"
cd "$HOME/Projects/replicating_RePO"

python cluster/repo_runner.py smoke --model flow \
  --results-root "$SCRATCH/replicating-repo-results/smoke"

python cluster/repo_runner.py smoke --model packet \
  --results-root "$SCRATCH/replicating-repo-results/smoke"
```

O smoke valida GPU, leitura dos dados, carregamento do SavedModel, pontuação e
um passo de ataque. Ele termina com `smoke test concluído`.

Se o smoke não couber em 30 minutos, submeta pela fila normal:

```bash
SMOKE=1 bash cluster/submit.sh smoke flow
SMOKE=1 bash cluster/submit.sh smoke packet
```

## 7. Avaliação normal (RePO e RePO+)

```bash
cd "$HOME/Projects/replicating_RePO"
bash cluster/submit.sh normal flow
bash cluster/submit.sh normal packet
```

Isso submete quatro jobs (dois modos por modelo). Acompanhe:

```bash
squeue -u "$USER"
```

Tempo esperado: poucos minutos por job. Resultados:

```text
$SCRATCH/replicating-repo-results/flow/normal/repo.json
$SCRATCH/replicating-repo-results/flow/normal/repo_plus.json
$SCRATCH/replicating-repo-results/packet/normal/repo.json
$SCRATCH/replicating-repo-results/packet/normal/repo_plus.json
```

Confira o threshold, o FPR e o TPR médio:

```bash
export RESULTS_ROOT="$SCRATCH/replicating-repo-results"
python - <<'PY'
import json
import os
from pathlib import Path

root = Path(os.environ["RESULTS_ROOT"])
for path in sorted(root.glob("*/normal/*.json")):
    data = json.loads(path.read_text())
    print(
        path,
        "threshold=", round(data["threshold"], 6),
        "fpr=", round(data["fpr"], 4),
        "fpr@paper=", data.get("fpr_at_paper_threshold"),
        "avg_tpr=", data.get("average_tpr"),
    )
PY
```

O campo `fpr_at_paper_threshold` mostra qual FPR o threshold adversarial do
artigo produz nos dados benignos. Ele deve ficar próximo de 0,10.

## 8. Piloto adversarial de fluxo

```bash
cd "$HOME/Projects/replicating_RePO"
bash cluster/submit.sh adversarial flow "DoS Slowhttptest"
```

Tempo esperado: cerca de 3 horas. Acompanhe:

```bash
squeue -u "$USER"
tail -f "$(ls -t "$SCRATCH/replicating-repo-results/logs/"*.out | head -n1)"
```

Resultado:

```text
$SCRATCH/replicating-repo-results/flow/adversarial/DoS-Slowhttptest.json
```

Confira:

```bash
python -m json.tool \
  "$SCRATCH/replicating-repo-results/flow/adversarial/DoS-Slowhttptest.json" | head -40
```

## 9. Piloto adversarial de pacotes

Esse é o job longo: cerca de 23 horas.

```bash
bash cluster/submit.sh adversarial packet "FTP-Patator"
```

Ele cabe na partição `short-simple` (2 dias). O runner salva checkpoint a cada
50 pacotes; se o job for interrompido, basta reenviar o mesmo comando.

```bash
squeue -u "$USER"
ls -lh "$SCRATCH/replicating-repo-results/packet/adversarial/checkpoints/"
```

Resultado:

```text
$SCRATCH/replicating-repo-results/packet/adversarial/FTP-Patator.json
```

## 10. Rodada completa

Depois de validar os pilotos:

```bash
cd "$HOME/Projects/replicating_RePO"

# 11 ataques de fluxo, 2 concorrentes
bash cluster/submit.sh adversarial flow all

# 11 ataques de pacotes, 2 concorrentes
bash cluster/submit.sh adversarial packet all
```

Os pilotos já concluídos são detectados pelo `config_hash` e ignorados; não há
custo em reenviar `all`.

Estimativas:

| Modelo | Ataques | Tempo por ataque | Com 2 concorrentes |
|---|---|---|---|
| fluxo | 11 | ~3 h | ~17 h |
| pacotes | 11 | ~23 h | ~6 dias |

Acompanhe o array:

```bash
squeue -u "$USER"
sacct -u "$USER" --format=JobID,JobName%30,State,Elapsed,MaxRSS
```

## 11. Se um job for interrompido

Reenvie o mesmo ataque:

```bash
bash cluster/submit.sh adversarial packet "FTP-Patator"
```

O runner carrega o checkpoint e continua do ponto salvo. Se a configuração
mudou (modelo, seed, threshold ou `max_samples`), ele recusa continuar. Use
`FORCE=1` somente se quiser descartar o checkpoint:

```bash
FORCE=1 bash cluster/submit.sh adversarial packet "FTP-Patator"
```

Para cancelar um job:

```bash
scancel <jobid>
```

## 12. Agregar e coletar os resultados

```bash
cd "$HOME/Projects/replicating_RePO"
bash cluster/submit.sh export
```

Isso gera, no repositório:

```text
results/cluster/summary.csv
results/cluster/summary.md
results/cluster/summary.json
```

Leia o resumo:

```bash
sed -n '1,160p' results/cluster/summary.md
```

Guarde os resultados brutos antes da limpeza de 90 dias do `$SCRATCH`:

```bash
mkdir -p "${ARCHIVE:-$HOME}/replicating-repo"
tar cJf "${ARCHIVE:-$HOME}/replicating-repo/results-$(date -u +%Y%m%d).tar.xz" \
  -C "$SCRATCH/replicating-repo-results" .
```

Para copiar para a máquina local, descubra o caminho absoluto no cluster:

```bash
echo "$SCRATCH/replicating-repo-results"
```

E, na máquina local:

```bash
rsync --bwlimit=1000 -azP \
  <login>@slurm-client1.cin.ufpe.br:<caminho-absoluto>/ \
  ~/Projects/replicating_RePO/replicating-repo-results/
```

## 13. Treino (opcional)

Para reproduzir também o treinamento:

```bash
bash cluster/submit.sh train flow
bash cluster/submit.sh train packet
```

Os modelos são gravados em:

```text
models/retrained/flw_model
models/retrained/pkt_model
```

Os modelos dos autores em `models/` nunca são sobrescritos. Para avaliar os
modelos retreinados:

```bash
MODEL_SOURCE=retrained bash cluster/submit.sh normal flow
MODEL_SOURCE=retrained bash cluster/submit.sh normal packet
MODEL_SOURCE=retrained bash cluster/submit.sh adversarial flow all
MODEL_SOURCE=retrained bash cluster/submit.sh adversarial packet all
```

O treino não tem checkpoint; se for interrompido, reenvie o job.

## 14. Checklist final

- [ ] branch `feat/apuana-cluster` no cluster;
- [ ] `cluster/setup_env.sh` concluído;
- [ ] smoke de fluxo e pacotes concluído;
- [ ] `normal flow` e `normal packet` concluídos;
- [ ] piloto adversarial de fluxo concluído;
- [ ] piloto adversarial de pacotes concluído;
- [ ] `adversarial flow all` concluído;
- [ ] `adversarial packet all` concluído;
- [ ] `summary.md` revisado;
- [ ] resultados arquivados/copiados;
- [ ] `docs/PROJECT_TASKS.md` e `report.tex` atualizados.

## 15. Problemas comuns

| Sintoma | Causa provável | Solução |
|---|---|---|
| `module: command not found` | shell não-login ou módulos não carregados | `source /etc/profile` ou abra um shell de login |
| `No module named tensorflow` | venv não ativado | `source "$SCRATCH/replicating-repo-venv/bin/activate"` |
| GPU vazia em `tf.config.list_physical_devices("GPU")` | job sem GPU ou rodando no login | submeta com `--gres=gpu:1` e verifique `nvidia-smi` |
| `FileNotFoundError` de CSV/`.npy` | dados não extraídos ou symlink quebrado | `bash cluster/setup_env.sh --force` |
| Job preso em `PD` | recursos acima do QoS da partição | use os defaults (8 CPU, 32G, 1 GPU) |
| Job morto por tempo | walltime insuficiente | reenvie; o checkpoint retoma; aumente `WALLTIME_ADV` |
| `checkpoint ... configuração diferente` | modelo/seed/threshold/max_samples mudou | use os mesmos parâmetros ou `FORCE=1` |
| Pouco espaço | dados + venv + resultados no `$SCRATCH` | `df -h "$SCRATCH"`, arquive e limpe resultados antigos |
| `pip install` falha no `tensorflow[and-cuda]` | wheel CUDA indisponível | use `tensorflow==2.21.0` e carregue o módulo CUDA do cluster |

## 16. Referências rápidas

- Runbook detalhado: [`../cluster/README.md`](../cluster/README.md)
- Checklist do projeto: [`PROJECT_TASKS.md`](PROJECT_TASKS.md)
- Requisitos do relatório: [`REPORT_REQUIREMENTS.md`](REPORT_REQUIREMENTS.md)
- Documentação oficial do Apuana: <https://apuana.cin.ufpe.br/usage>
