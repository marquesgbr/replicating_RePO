# replicating_RePO

Reprodução e extensão dos experimentos de robustez contra exemplos adversariais em Sistemas de Detecção de Intrusão de Rede (NIDS) baseados em autoencoders (*RePO* e *RePO+*), incluindo avaliação em um conjunto de dados adicional (CIC-ToN-IoT) e propostas de redução do custo de inferência por alocação dinâmica de máscaras por amostra.

- **Artigo original:** [Enhancing Robustness Against Adversarial Examples in Network Intrusion Detection Systems](https://arxiv.org/abs/2008.03677)
- **Repositório original:** [s-mohammad-hashemi/repo](https://github.com/s-mohammad-hashemi/repo)

---

## Visão Geral e Stack Tecnológica

O projeto reproduz o protocolo *RePO/RePO+* nos cenários baseados em **fluxos** e **pacotes** do CICIDS2017, estende a avaliação para o dataset **CIC-ToN-IoT** (`data/new_data/`) e propõe duas regras de alocação de máscaras por amostra para reduzir o custo de inferência do *RePO+* (**cascata *cheap-first*** e **decisão sequencial** com parada antecipada por intervalo de confiança).

| Componente | Especificação |
| --- | --- |
| **Python (Local)** | `3.12.3` (WSL2 com GPU) / `3.13.1` (Windows nativo CPU) |
| **Python (VS Code Colab Extension)** | `3.13.16` (runtime remoto com GPU) |
| **Python (Cluster Apuana)** | `3.10.12` (`/usr/bin/python3.10` nos nós de computação) |
| **Deep Learning** | TensorFlow `2.21.0`; Keras `3.15.1` (Local/Colab) e `3.12.4` (Apuana) |
| **Dados e Testes** | `numpy`, `pandas`, `pytest`, `jupyterlab`, `ipykernel` (`requirements.txt`) |
| **Versionamento** | Git e Git LFS (usado em `data/new_data/New-dataset-split-clean.zip`) |
| **Relatório** | LaTeX (`IEEEtran`), compilado via `latexmk` (`report.tex` / `report_template.tex`) |
|      **IDE** | VSCode|

---

## Os 3 Pipelines Oficiais (Setup e Execução)

O projeto suporta três ambientes oficiais de execução:

| Pipeline | Python | Aceleração | Uso Indicado | Guia Detalhado |
| --- | --- | --- | --- | --- |
| **1. Execução Local** | `3.12.3` (WSL2) / `3.13.1` (Win) | GPU (WSL2) ou CPU | Desenvolvimento, testes unitários, *smoke tests* e avaliações locais | [GPU_TENSORFLOW_COMPATIBILITY.md](./docs/GPU_TENSORFLOW_COMPATIBILITY.md) |
| **2. Colab no VS Code** | `3.13.16` | GPU (Colab) | Execução de notebooks com GPU sem configurar CUDA localmente | [Colab_Setup.ipynb](./code/Colab_Setup.ipynb) |
| **3. Cluster Apuana** | `3.10.12` | GPU A100 80 GB | Rodadas completas (normal e adversarial) via jobs SLURM | [APUANA_RUNBOOK.md](./docs/APUANA_RUNBOOK.md) · [cluster/README.md](./cluster/README.md) |

### 1. Execução Local (GPU via WSL2 ou CPU)

Na raiz do repositório:

```bash
python -m venv .venv
# Linux / macOS / WSL2 / Git Bash:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name replicating-repo --display-name "Python (replicating_RePO)"
```

> **GPU no Windows:** o TensorFlow oficial (`>= 2.11`) não oferece suporte CUDA nativo no Windows. Para usar GPU localmente no Windows, utilize **WSL2** com `tensorflow[and-cuda]` e o kernel `Python (replicating_RePO - GPU)`, conforme descrito em [GPU_TENSORFLOW_COMPATIBILITY.md](./docs/GPU_TENSORFLOW_COMPATIBILITY.md).

### 2. Execução via Extensão do Colab no VS Code (GPU)

1. Instale as extensões **Python** (`ms-python.python`), **Jupyter** (`ms-toolsai.jupyter`) e **Google Colab** (`google.colab`) no VS Code.
2. Conecte um notebook a um runtime Colab com GPU pelo seletor de kernel (ou *Command Palette*: `Colab: Connect to Colab Runtime`).
3. Abra e execute [code/Colab_Setup.ipynb](./code/Colab_Setup.ipynb) em ordem para validar a GPU, clonar/atualizar o repositório, extrair os dados e testar o carregamento dos modelos.
4. Não execute `pip install -r requirements.txt` no Colab para não sobrescrever o TensorFlow pré-instalado do runtime.

### 3. Execução no Cluster Apuana (GPU A100)

Nos nós de login do cluster Apuana (CIn/UFPE, com VPN ativa):

```bash
cd "$HOME/Projects/replicating_RePO"
git switch main
bash cluster/setup_env.sh
bash cluster/submit.sh normal flow
bash cluster/submit.sh adversarial flow "DoS Slowhttptest"
bash cluster/submit.sh adversarial packet all
bash cluster/submit.sh export
```

Consulte [docs/APUANA_RUNBOOK.md](./docs/APUANA_RUNBOOK.md) para o roteiro linear completo e [cluster/README.md](./cluster/README.md) para detalhes técnicos de partições, checkpoints e agregação.

---

## Ordem Recomendada de Execução

Para compreender e reproduzir o fluxo completo do pipeline, siga esta sequência:

1. **Preparação dos dados** (a partir da raiz do repositório):
   ```bash
   bash extract_flow_data.sh
   bash extract_packet_data.sh
   bash extract_new_data.sh
   ```
   *(No Windows, execute via Git Bash ou WSL2. Os metadados de normalização em `data/*/x_train_meta/` já estão versionados.)*
2. **Setup do ambiente** (quando aplicável): `code/Colab_Setup.ipynb` (Colab VS Code) ou `cluster/setup_env.sh` (Apuana).
3. **Treinamento** *(opcional — os modelos pré-treinados já estão em `models/`)*: `code/train_flow_based_model.py`, `code/train_packet_based_model.py`, `code/train_new_data_model.py` (ou `cluster/train.py`).
4. **Avaliação Normal** *(sempre antes da Adversarial para estabelecer thresholds e referência)*:
   - Fluxos: `code/Flow_based_RePO_Normal_Test.ipynb`
   - Pacotes: `code/Packet_based_RePO_Normal_Test.ipynb`
   - Novo dataset: `code/new_data_RePO_Normal_Test.ipynb` (ou `PYTHONPATH=code python -m new_data_eval normal`)
5. **Avaliação Adversarial**:
   - Fluxos: `code/Flow_based_RePO_Adversarial.ipynb`
   - Pacotes: `code/Packet_based_RePO_Adversarial.ipynb`
   - Novo dataset: `code/new_data_RePO_Adversarial.ipynb` (ou `PYTHONPATH=code python -m new_data_eval adversarial`)
6. **Experimentos de Eficiência e Alocação de Máscaras** *(proposta da equipe)*:
   - Execução no cluster via `bash cluster/run_mask_study.sh` ou etapas individuais com `bash cluster/submit.sh mask-exp <etapa>` (`baseline`, `variance`, `cascade`, `cascade-attack`, `distill`, `distill-attack`, `sequential-attack`, `packet`, `aggregate`), ou localmente via `PYTHONPATH=code python -m mask_experiments.<módulo>`.
7. **Agregação e Figuras**: `cluster/aggregate_results.py`, `cluster/aggregate_new_data.py`, `code/mask_experiments/aggregate.py` e `code/generate_report_figures.py`.

---

## Novo Dataset (`data/new_data/`)

O dataset adicional (**CIC-ToN-IoT** — Sarhan et al., *Big Data Research*, vol. 30, p. 100359, 2022) avalia a generalização do protocolo *RePO/RePO+* em 77 features de fluxo. Os CSVs já processados (`new_train.csv` e `new_test.csv`) estão em `data/new_data/New-dataset-split-clean.zip` (Git LFS), com avaliação implementada no pacote `code/new_data_eval/` e nos notebooks *wrappers* `code/new_data_RePO_Normal_Test.ipynb` e `code/new_data_RePO_Adversarial.ipynb`.

Com o ambiente configurado e os dados extraídos (`bash extract_new_data.sh`), valide a integridade com:

```bash
PYTHONPATH=code python -m new_data_eval validate
```

Para o passo a passo completo de **como executar os testes do novo dataset em cada um dos 3 ambientes** (Local, Colab no VS Code e Cluster Apuana), consulte [docs/NEW_DATASET.md](./docs/NEW_DATASET.md).

---

## Melhoria Proposta: Alocação de Máscaras por Amostra (`code/mask_experiments/`)

Para mitigar o custo de 100 passagens por amostra do *RePO+*, a equipe implementou e avaliou duas regras de alocação dinâmica de máscaras em `code/mask_experiments/` (junto às direções comparativas de redução estática e destilação):

- **Proposta 1 — Cascata *cheap-first*** ([`cascade.py`](./code/mask_experiments/cascade.py), [`cascade_attack.py`](./code/mask_experiments/cascade_attack.py)): resolve amostras benignas inequívocas com um prefixo barato de máscaras ($n_{\text{cheap}} \in \{5, 10, 25\}$) e encaminha apenas as amostras ambíguas ao *RePO+* completo de 100 máscaras.
- **Proposta 2 — Decisão sequencial** ([`decision.py`](./code/new_data_eval/decision.py), [`sequential_attack.py`](./code/mask_experiments/sequential_attack.py)): revela máscaras em blocos e interrompe a inferência quando o intervalo de confiança de 95% da média amostral exclui o limiar $\tau$, além de avaliar o ataque adversarial direcionado ao limite superior desse intervalo (`--loss ci` em [`attack.py`](./code/new_data_eval/attack.py)).
- **Direções comparativas:** curva de variância e redução estática ([`variance.py`](./code/mask_experiments/variance.py), [`adversarial.py`](./code/mask_experiments/adversarial.py)), destilação em aluno de passe único ([`distill.py`](./code/mask_experiments/distill.py), [`distill_attack.py`](./code/mask_experiments/distill_attack.py)) e confirmação em pacotes ([`packet.py`](./code/mask_experiments/packet.py)).

Para executar os experimentos localmente (a partir da raiz, com `.venv` ativo) ou no Cluster Apuana:

```bash
# Execução local de uma etapa (ex.: cascata ou ataque sequencial)
PYTHONPATH=code python -m mask_experiments.cascade
PYTHONPATH=code python -m mask_experiments.sequential_attack --help

# No Cluster Apuana (cadeia completa ou etapa isolada)
bash cluster/run_mask_study.sh
bash cluster/submit.sh mask-exp cascade
bash cluster/submit.sh mask-exp cascade-attack all
bash cluster/submit.sh mask-exp sequential-attack all
```

As análises detalhadas estão documentadas em [`docs/MASK_EFFICIENCY_CASCADE.md`](./docs/MASK_EFFICIENCY_CASCADE.md), [`docs/MASK_EFFICIENCY_SEQUENTIAL.md`](./docs/MASK_EFFICIENCY_SEQUENTIAL.md), [`docs/MASK_EFFICIENCY_RESULTS.md`](./docs/MASK_EFFICIENCY_RESULTS.md), [`docs/MASK_EFFICIENCY_INTERIM.md`](./docs/MASK_EFFICIENCY_INTERIM.md) e [`docs/MASK_EFFICIENCY_RUNS.md`](./docs/MASK_EFFICIENCY_RUNS.md).

---

## Reprodução dos Modelos Treinados

Os modelos já treinados estão em `models/flw_model/`, `models/pkt_model/` e `models/new_data_model/` no formato *SavedModel* legado do TensorFlow. Como o Keras 3 não carrega esse formato diretamente via `load_model()`, os notebooks e scripts utilizam o adaptador [`code/load_model_compat.py`](./code/load_model_compat.py), preservando os gradientes necessários aos ataques adversariais.

Escolha o procedimento correspondente ao seu ambiente:

### No Ambiente Local (GPU via WSL2 ou CPU)

1. Ative o ambiente virtual (`.venv`) e abra o VS Code (na janela `WSL: Ubuntu` se estiver usando GPU) ou inicie o JupyterLab a partir de `code/`:
   ```bash
   source .venv/bin/activate
   cd code
   python -m jupyter lab
   ```
2. Selecione o kernel `Python (replicating_RePO - GPU)` (WSL2) ou `Python (replicating_RePO)` (CPU).
3. Execute as células em ordem, iniciando pelos notebooks `Normal` antes dos `Adversarial`.

### Na Extensão do Colab no VS Code (GPU)

1. Conecte o VS Code a um runtime Colab com GPU.
2. Execute [code/Colab_Setup.ipynb](./code/Colab_Setup.ipynb) até o fim (garante `/content/replicating_RePO/code` no `sys.path`, GPU ativa e dados extraídos).
3. Abra e execute os notebooks de avaliação (`Normal` antes de `Adversarial`) no mesmo runtime.

### No Cluster Apuana (GPU A100)

1. No nó de login, após `bash cluster/setup_env.sh`, valide com um *smoke test* ou submeta os jobs SLURM diretamente:
   ```bash
   bash cluster/submit.sh normal flow
   bash cluster/submit.sh normal packet
   bash cluster/submit.sh adversarial flow all
   bash cluster/submit.sh adversarial packet all
   ```
2. Para o novo dataset:
   ```bash
   bash cluster/submit.sh new-data normal
   bash cluster/submit.sh new-data adversarial all primary
   ```
3. Para os experimentos da melhoria proposta (alocação e eficiência de máscaras):
   ```bash
   bash cluster/run_mask_study.sh
   bash cluster/submit.sh mask-exp cascade
   bash cluster/submit.sh mask-exp cascade-attack all
   bash cluster/submit.sh mask-exp sequential-attack all
   ```

---

## Resultados e Relatório

Para consultar os **resultados consolidados, tabelas comparativas, análises estatísticas e discussões**, consulte primariamente o relatório final em LaTeX ([`report.tex`](./report.tex) e [`report.pdf`](./report.pdf)).

Caso deseje inspecionar os **artefatos brutos** produzidos pelas execuções:

- **Pasta `results/`:**
  - Raiz de `results/`: contém os `.json` gerados pelas execuções locais e via extensão do **Colab no VS Code** (`flow_based_adversarial_results.json`, `packet_based_adversarial_results.json`).
  - Subpasta [`results/mask_efficiency/`](./results/mask_efficiency/): contém os `.json` dos experimentos da melhoria proposta pela equipe (`cascade_flow.json`, `cascade_attack.json`, `variance_flow.json`, `distillation_flow.json`, `distill_attack.json`, `adversarial_pilot.json`, `packet_normal.json`, `baseline_flow_smoke.json`).
- **Documentos de análise em `docs/`:** consolidam os resultados executados no **Cluster Apuana** (cujos scripts de execução e agregação estão em `cluster/` e `code/mask_experiments/`):
  - [`docs/REPLICATION_ANALYSIS.md`](./docs/REPLICATION_ANALYSIS.md) — replicação em fluxos e pacotes (CICIDS2017).
  - [`docs/NEW_DATASET_ANALYSIS.md`](./docs/NEW_DATASET_ANALYSIS.md) — avaliação no novo dataset (CIC-ToN-IoT).
  - [`docs/MASK_EFFICIENCY_CASCADE.md`](./docs/MASK_EFFICIENCY_CASCADE.md) e [`docs/MASK_EFFICIENCY_SEQUENTIAL.md`](./docs/MASK_EFFICIENCY_SEQUENTIAL.md) — resultados e análises das propostas de cascata *cheap-first* e decisão sequencial.
  - [`docs/MASK_EFFICIENCY_RESULTS.md`](./docs/MASK_EFFICIENCY_RESULTS.md), [`docs/MASK_EFFICIENCY_INTERIM.md`](./docs/MASK_EFFICIENCY_INTERIM.md) e [`docs/MASK_EFFICIENCY_RUNS.md`](./docs/MASK_EFFICIENCY_RUNS.md) — consolidação geral, achados intermediários e matriz de execuções do estudo de máscaras.

---

## Escrita e Compilação do Relatório

O documento principal é [`report.tex`](./report.tex), estruturado a partir de [`report_template.tex`](./report_template.tex) e dos requisitos em [`docs/REPORT_REQUIREMENTS.md`](./docs/REPORT_REQUIREMENTS.md).
É necessário instalar uma distribuição LaTeX com `latexmk` no `PATH` (**TeX Live** funciona bem no Linux, Windows e macOS, **MiKTeX** no Windows ou **MacTeX** no macOS) e os pacotes `IEEEtran`, `graphicx`, `amsmath`, `algorithm` e `algpseudocode`.

### Compilação pelo VS Code

1. Instale a extensão **LaTeX Workshop** (`james-yu.latex-workshop`).
3. Abra `report.tex` e execute `Build LaTeX project` (`Ctrl+Alt+B`).

### Compilação manual via terminal

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

*(Arquivos auxiliares como `.aux`, `.log`, `.fls`, `.fdb_latexmk` e `.out` são produtos locais de compilação.)*

Alternativamente, se não quiser instalar nada na máquina, pode-se compilar usando serviços online como Overleaf.

---

## Estrutura Relevante do Repositório

```text
replicating_RePO/
├── README.md
├── requirements.txt
├── extract_flow_data.sh
├── extract_packet_data.sh
├── extract_new_data.sh
├── report.tex
├── report.pdf
├── report_template.tex
├── images/                             # Figuras geradas para o relatório
├── cluster/                            # Scripts SLURM, runner, treino e agregação no Apuana
│   ├── README.md
│   ├── requirements-cluster.txt
│   ├── setup_env.sh
│   ├── submit.sh
│   ├── run_mask_study.sh
│   ├── slurm/
│   ├── train.py
│   ├── repo_runner.py
│   ├── metrics.py
│   ├── aggregate_results.py
│   ├── aggregate_new_data.py
│   ├── analyze_adversarial.py
│   ├── analyze_sampling.py
│   ├── lfs_fetch.py
│   └── tests/
├── code/                               # Notebooks, adaptadores, treino e pacotes de avaliação
│   ├── Colab_Setup.ipynb
│   ├── Flow_based_RePO_Normal_Test.ipynb
│   ├── Flow_based_RePO_Adversarial.ipynb
│   ├── Packet_based_RePO_Normal_Test.ipynb
│   ├── Packet_based_RePO_Adversarial.ipynb
│   ├── new_data_RePO_Normal_Test.ipynb
│   ├── new_data_RePO_Adversarial.ipynb
│   ├── load_model_compat.py
│   ├── experiment_results.py
│   ├── train_flow_based_model.py
│   ├── train_packet_based_model.py
│   ├── train_new_data_model.py
│   ├── generate_report_figures.py
│   ├── mask_experiments/               # Experimentos de alocação e eficiência de máscaras
│   └── new_data_eval/                  # Pacote de avaliação e testes do novo dataset
├── data/
│   ├── flow_based/                     # ZIPs CICIDS2017 (fluxo) e x_train_meta/
│   ├── packet_based/                   # ZIPs CICIDS2017 (pacotes) e x_train_meta/
│   └── new_data/                       # ZIP LFS (CIC-ToN-IoT) e x_train_meta/
├── docs/
│   ├── GPU_TENSORFLOW_COMPATIBILITY.md
│   ├── APUANA_RUNBOOK.md
│   ├── NEW_DATASET.md
│   ├── REPLICATION_ANALYSIS.md
│   ├── NEW_DATASET_ANALYSIS.md
│   ├── MASK_EFFICIENCY_CASCADE.md
│   ├── MASK_EFFICIENCY_SEQUENTIAL.md
│   ├── MASK_EFFICIENCY_RESULTS.md
│   ├── MASK_EFFICIENCY_INTERIM.md
│   ├── MASK_EFFICIENCY_RUNS.md
│   ├── PROJECT_TASKS.md
│   └── REPORT_REQUIREMENTS.md
├── models/
│   ├── flw_model/
│   ├── pkt_model/
│   └── new_data_model/
└── results/                            # JSONs das execuções locais, Colab VS Code e mask_efficiency/
    └── mask_efficiency/                # JSONs dos experimentos de eficiência de máscaras
```
