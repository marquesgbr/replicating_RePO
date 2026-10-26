# replicating_RePO

Reprodução de experimentos sobre melhoria da robustez contra exemplos
adversariais em sistemas de detecção de intrusão de rede usando autoencoders.

- Artigo original: [Enhancing Robustness Against Adversarial Examples in
  Network Intrusion Detection Systems](https://arxiv.org/abs/2008.03677)
- Repositório original: [s-mohammad-hashemi/repo](https://github.com/s-mohammad-hashemi/repo)

## Estrutura relevante

- `code/`: notebooks de avaliação e scripts de treinamento.
- `docs/`: documentação de compatibilidade, execução no Colab e requisitos do
  relatório, além do [checklist do projeto](./docs/PROJECT_TASKS.md).
- `data/flow_based/` e `data/packet_based/`: arquivos compactados, dados
  extraídos e metadados de normalização.
- `models/flw_model/` e `models/pkt_model/`: modelos já treinados usados pelos
  notebooks.
- `extract_flow_data.sh` e `extract_packet_data.sh`: scripts de preparação dos
  dados.
- `report.tex`: relatório em LaTeX.

## Requisitos

O TensorFlow pode exigir uma versão específica do Python e do sistema
operacional. Para os procedimentos de compatibilidade com GPU, consulte
[gpu_tensorflow_compatibility.md](./docs/gpu_tensorflow_compatibility.md).

Em resumo:

- **Linux/WSL2:** é a opção recomendada para usar uma GPU NVIDIA com versões
  atuais do TensorFlow. No WSL2, confirme `nvidia-smi`, instale o TensorFlow
  com `tensorflow[and-cuda]` e valide `tf.config.list_physical_devices("GPU")`.
- **Windows nativo:** o TensorFlow oficial deixou de oferecer suporte CUDA
  nativo a partir da versão 2.11. O TensorFlow 2.10 ainda pode ser usado com
  CUDA/cuDNN compatíveis, mas exige um ambiente separado e uma versão de Python
  suportada por essa versão antiga; não é compatível com o Python 3.12/3.13 usado
  no ambiente atual do projeto.
- **Google Colab no VS Code:** instale a extensão Google Colab, conecte o
  notebook a um runtime Colab com GPU pelo seletor de kernel e execute
  [code/Colab_Setup.ipynb](./code/Colab_Setup.ipynb). Evite substituir
  automaticamente o TensorFlow pré-instalado executando
  `pip install -r requirements.txt`.
- **Google Colab no navegador:** quando não houver GPU local, use o menu
  `Runtime > Change runtime type > GPU`, confirme a GPU com `nvidia-smi` e
  siga o guia [colab_browser.md](./docs/colab_browser.md).

Se o TensorFlow for executado apenas na CPU, escolha uma versão de Python
compatível com a versão instalada e verifique se o kernel do notebook aponta
para o mesmo ambiente virtual. O documento de compatibilidade também explica
como configurar as bibliotecas CUDA/cuDNN e como diagnosticar kernels que não
herdam o ambiente correto.

### Stack do projeto

#### Software e bibliotecas

- Python:
  - Python 3.13.1 no ambiente nativo Windows usado durante o diagnóstico;
  - Python 3.12.3 no ambiente WSL2 usado com GPU;
  - use a versão compatível com o TensorFlow escolhido no ambiente.
- TensorFlow 2.21.0 e Keras 3.15.1 no ambiente de referência atual;
- NumPy e pandas para manipulação dos dados;
- JupyterLab e `ipykernel` para execução dos notebooks;
- Git para versionamento e clonagem do repositório;
- LaTeX, `latexmk` e os pacotes do template para compilação do relatório.

As versões instaladas no ambiente podem ser conferidas com:

```bash
python --version
python -m pip freeze
```

No WSL2 com GPU, o TensorFlow utiliza as bibliotecas NVIDIA instaladas com
`tensorflow[and-cuda]`. O driver NVIDIA do Windows precisa estar instalado e
`nvidia-smi` deve funcionar dentro do WSL2.

#### IDE

VSCode

#### Extensões do VS Code

As extensões abaixo cobrem os fluxos de desenvolvimento, notebooks, WSL2,
Colab e relatório:

| Extensão | Identificador | Uso |
| --- | --- | --- |
| Python | `ms-python.python` | Interpretador, ambientes virtuais e execução Python |
| Jupyter | `ms-toolsai.jupyter` | Execução e seleção de kernels nos notebooks |
| WSL | `ms-vscode-remote.remote-wsl` | Abrir o projeto no Linux do WSL2 com `code .` |
| Colab | `google.colab` | Abrir e executar notebooks no fluxo do Colab |
| LaTeX Workshop | `james-yu.latex-workshop` | Compilar e visualizar `report.tex` |

Instale Python e Jupyter para trabalhar com os notebooks localmente. A
extensão WSL é necessária apenas para o fluxo Windows + WSL2. A extensão
Google Colab é opcional quando os notebooks forem executados diretamente na
interface web do Colab. LaTeX Workshop é necessária apenas para editar e
compilar o relatório.

Extensões de Git, como GitLens ou GitHub Pull Requests and Issues, podem ser
úteis para colaboração, mas não são necessárias para executar ou reproduzir
os experimentos.

### Ambiente Python

Se já possuir um ambiente Python que atenda à `requirements.txt`, pode pular
esta parte, ativá-lo e seguir para a seção de execução. Para GPU no Windows,
use o fluxo WSL2 descrito em
[gpu_tensorflow_compatibility.md](./docs/gpu_tensorflow_compatibility.md), em vez
de instalar o TensorFlow atual no Python nativo do Windows.

Na raiz do repositório:

```bash
python -m venv .venv
```


Ative o ambiente:

```bash
# Linux, macOS, Git Bash ou WSL
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1
```

Instale as dependências:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m ipykernel install --user --name replicating-repo --display-name "Python (replicating_RePO)"
```

Em um ambiente local comum, os notebooks usam o kernel
`Python (replicating_RePO)`. No WSL2 com GPU, siga a instalação específica do
documento de compatibilidade e use o kernel `Python (replicating_RePO - GPU)`.
As versões instaladas podem ser registradas para diagnóstico com:

```bash
python -m pip freeze
```

## Preparação dos dados

Os scripts `.sh` devem ser executados a partir da raiz do repositório, pois
usam caminhos relativos. Em Linux ou macOS:

```bash
bash extract_flow_data.sh
bash extract_packet_data.sh
```

No Windows, execute os mesmos comandos pelo Git Bash ou pelo WSL. O WSL é a
opção mais previsível para `bash`, `cat` e `unzip`; nesse caso, abra a pasta do
repositório pelo caminho montado em `/mnt/<unidade>/...`. Não execute os
scripts em `code/`, pois os caminhos relativos deixariam de apontar para os
dados.

Os scripts extraem os arquivos de fluxo e remontam as partes dos arquivos de
pacotes antes de extraí-los. Essa etapa só precisa ser repetida quando os
dados extraídos forem removidos ou quando os arquivos compactados forem
substituídos. Confirme antes que as pastas de dados correspondentes foram
criadas e que contêm os arquivos esperados. Os arquivos `.npy` em
`data/*/x_train_meta/` são metadados já versionados e são usados na
normalização durante a avaliação.

## Reprodução com os modelos já treinados

O procedimento abaixo avalia os modelos presentes em `models/`; não é
necessário treinar novamente. Os caminhos usados pelos notebooks são
relativos à pasta `code/`.

### Execução local no Linux, macOS ou Git Bash

Ative o ambiente e inicie o Jupyter a partir de `code/`:

```bash
source .venv/bin/activate
cd code
python -m jupyter lab
```

No VS Code, abra um notebook em `code/` e selecione o kernel
`Python (replicating_RePO)`. Execute as células em ordem, começando pelo
início, depois de reiniciar o kernel. Evite executar células de otimização
repetidamente no mesmo kernel, pois elas podem alterar variáveis e resultados
intermediários.

No Windows nativo, ative o ambiente pelo PowerShell antes de iniciar o
Jupyter:

```powershell
.\.venv\Scripts\Activate.ps1
Set-Location code
python -m jupyter lab
```

Esse fluxo é adequado para CPU. Para GPU, use WSL2.

### Execução no Windows com WSL2 e GPU

Com o ambiente `.venv` configurado dentro do WSL2:

```bash
cd /mnt/d/Projects/26.2/TADI/replicating_RePO
source .venv/bin/activate
code .
```

Na janela do VS Code conectada ao WSL2, abra o notebook e selecione
`Python (replicating_RePO - GPU)`. Esse fluxo mostra o kernel diretamente e
evita copiar a URL de um servidor Jupyter. Os detalhes de CUDA, cuDNN,
`LD_LIBRARY_PATH` e a alternativa de conexão por servidor estão em
[gpu_tensorflow_compatibility.md](./docs/gpu_tensorflow_compatibility.md).

### Execução no Google Colab pelo VS Code

Instale as extensões Python, Jupyter e Google Colab. Abra o notebook no VS
Code, use o seletor de kernel para conectar a um runtime hospedado pelo Colab
e escolha um runtime com GPU. O nome exato dos comandos pode variar conforme
a versão da extensão; procure por comandos como **Colab: Connect to Colab
Runtime** na Command Palette.

Depois de conectar:

1. abra [code/Colab_Setup.ipynb](./code/Colab_Setup.ipynb);
2. execute as células em ordem;
3. confirme que `tf.config.list_physical_devices("GPU")` retorna uma GPU;
4. execute os notebooks de avaliação no mesmo runtime.

Nesse fluxo, os comandos Python são executados remotamente no Colab, e não no
Python local nem no WSL2. A extensão Google Colab gerencia a conexão; não é
necessário iniciar `jupyter lab`, copiar URL ou configurar `LD_LIBRARY_PATH`.
Consulte [gpu_tensorflow_compatibility.md](./docs/gpu_tensorflow_compatibility.md)
para detalhes de diagnóstico.

### Execução no Google Colab pelo navegador

Para usar a interface web, siga [colab_browser.md](./docs/colab_browser.md). Esse
fluxo usa os menus `Runtime` do Colab, que não fazem parte do fluxo conectado
ao VS Code.

### Ordem recomendada

Execute os notebooks `Normal` antes dos `Adversarial`, tanto para o fluxo
baseado em pacotes quanto para o baseado em fluxos. Os notebooks normais
estabelecem a referência da avaliação; os notebooks adversariais produzem os
resultados para comparação.

Preserve as saídas geradas pela execução se elas forem necessárias para o
relatório, mas registre no relatório o notebook, o kernel, a data e o commit
usados. A execução pode consumir bastante memória e tempo, especialmente no
cenário adversarial; feche kernels que não estiverem em uso.

Os diretórios em `models/` estão no formato SavedModel legado do TensorFlow.
Como o Keras 3 não aceita esse formato diretamente em `load_model()`, os
notebooks usam `code/load_model_compat.py`. Esse adaptador também converte a
saída única da assinatura `serving_default` para um tensor, mantendo o cálculo
de gradientes necessário nos notebooks adversariais. Execute os notebooks a
partir de `code/` para que o módulo seja encontrado.

### Execução por linha de comando (opcional; Não recomendado)

Para executar um notebook sem abrir a interface gráfica, a instalação de
Jupyter inclui o `nbconvert`:

```bash
cd code
python -m jupyter nbconvert --to notebook --execute \
  --ExecutePreprocessor.timeout=-1 \
  --output-dir=executed \
  Flow_based_RePO_Normal_Test.ipynb
```

Repita trocando o nome do notebook. A pasta `code/executed/` é apenas uma
pasta de saída local e não deve ser versionada sem necessidade.

## Escrita e compilação do relatório

O arquivo a editar é `report.tex`; `report_template.tex` é uma referência.
Os requisitos detalhados estão em
[docs/requisitos_relatório.md](./docs/requisitos_relatório.md). Não há um
`report.txt` neste repositório. Para editar e compilar pelo VS
Code:

1. Instale a extensão **LaTeX Workshop** (`James-Yu.latex-workshop`).
2. Instale uma distribuição LaTeX:
   - Windows: **MiKTeX** ou **TeX Live** (Recomendado);
   - Linux: **TeX Live** com os pacotes extras necessários;
   - macOS: **MacTeX**.
3. Garanta que `latexmk` esteja disponível no `PATH`. No Windows, confirme
   também que a distribuição escolhida instalou os executáveis auxiliares
   necessários.
4. Instale ou habilite os pacotes usados pelo documento: `IEEEtran`,
   `graphicx`, `amsmath`, `algorithm` e `algpseudocode`. O `IEEEtran` e os
   pacotes de algoritmo podem exigir instalação adicional em uma instalação
   mínima. No MiKTeX, habilitar a instalação automática de pacotes é a opção
   mais simples; no TeX Live, instale-os pelo `tlmgr` ou pelo gerenciador da
   distribuição.
5. Abra `report.tex` e use `Build LaTeX project` na barra do LaTeX Workshop
   ou `Ctrl+Alt+B`. A receita padrão com `latexmk` normalmente é suficiente;
   se houver referências BibTeX no futuro, use a receita que inclua BibTeX.

[Guia Tutorial](https://github.com/James-Yu/LaTeX-Workshop/wiki/Install) para instalação de requerimentos do LaTeX Workshop

Uma compilação manual equivalente, a partir da raiz, é:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error report.tex
```

Se o build falhar por pacote ausente, instale o pacote na distribuição LaTeX,
em vez de copiar arquivos `.sty` para o repositório. Arquivos auxiliares
(`.aux`, `.log`, `.fls`, `.fdb_latexmk`, `.out` e o PDF) são produtos locais
da compilação e não devem ser incluídos no commit, salvo se o projeto decidir
explicitamente versionar o PDF final.

## Boas práticas para colaborar

- Faça alterações pequenas e focadas; não reformate todo o `report.tex` ao
  alterar uma seção.
- Use finais de linha e codificação consistentes e evite alterar espaços,
  indentação ou quebras de linha sem necessidade.
- Prefira editar o `.tex` como texto e gere o PDF localmente. Não faça
  commit dos arquivos auxiliares do LaTeX.
- Antes de enviar mudanças, confira `git diff --check` e `git diff` e
  confirme que apenas os arquivos pretendidos foram alterados.
- Faça `git pull --rebase` antes de começar a escrever, e envie commits
  pequenos com mensagens descritivas. Isso reduz conflitos e torna o
  histórico do relatório mais fácil de revisar.
- Ao adicionar resultados, identifique a origem (notebook, modelo, dados,
  kernel e commit) e evite substituir resultados anteriores sem registrar a
  mudança.
- Não salve credenciais, caminhos absolutos da sua máquina ou grandes saídas
  de notebook no repositório. Se uma saída for essencial, mantenha apenas a
  saída mínima e reproduzível.
