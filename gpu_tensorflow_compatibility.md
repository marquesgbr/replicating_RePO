# Compatibilidade do TensorFlow com GPU

Este documento descreve a compatibilidade do TensorFlow com GPU neste projeto,
a versão de Python utilizada e uma alternativa para executar os notebooks no
Google Colab quando o computador local não possui uma GPU compatível.

## Ambiente utilizado no projeto

A partir do TensorFlow 2.11, o TensorFlow oficial deixou de oferecer
suporte CUDA nativo para Windows. Por isso, instalar apenas o driver NVIDIA,
CUDA ou cuDNN no Windows não é suficiente para habilitar a GPU nas versões
atuais do TensorFlow.

Esse diagnóstico pode ser confirmado com:

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Build com CUDA:", tf.test.is_built_with_cuda())
print("GPUs físicas:", tf.config.list_physical_devices("GPU"))
print("GPUs lógicas:", tf.config.list_logical_devices("GPU"))
```

Quando o TensorFlow não consegue utilizar a GPU, o resultado normalmente é:

```text
Build com CUDA: False
GPUs físicas: []
GPUs lógicas: []
```

Nesse caso, as operações do TensorFlow serão executadas na CPU.

## Alternativas para usar GPU localmente

### WSL2

Como o TensorFlow oficial não oferece suporte CUDA nativo no Windows a partir
da versão 2.11, a opção recomendada para Windows é executar o projeto em uma
distribuição Linux por meio do WSL2. Nesse cenário, o TensorFlow pode utilizar
a GPU NVIDIA através do suporte CUDA para Linux. Essa limitação é do
TensorFlow no Windows.

No powershell como administrador:

```bash
wsl --install -d Ubuntu
```

Há também a opção de instalar o Ubuntu pela Microsoft Store.


Depois de instalar e configurar o WSL2, confirme a GPU dentro do Linux:

```bash
nvidia-smi
```

Em seguida, crie um ambiente virtual dentro do WSL2 e instale o TensorFlow:

```bash
cd /mnt/d/Projects/26.2/TADI/replicating_RePO

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install --upgrade "tensorflow[and-cuda]"
python -m ipykernel install --user \
  --name replicating-repo-gpu \
  --display-name "Python (replicating_RePO - GPU)"
```

Como `requirements.txt` contém o requisito genérico `tensorflow`, a instalação
com `tensorflow[and-cuda]` deve ser feita por último. Assim, a versão com as
dependências NVIDIA permanece instalada.

Em alguns ambientes, os wheels NVIDIA ficam instalados no ambiente virtual, mas o loader de bibliotecas do Linux não inclui automaticamente esses diretórios. Nesse caso, configure o caminho antes de iniciar o Jupyter ou executar o Python:

```bash
V="$PWD/.venv"
export LD_LIBRARY_PATH="$(find "$V/lib/python3.12/site-packages/nvidia" \
  -type d -name lib -printf '%p:' | sed 's/:$//')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

O caminho `python3.12` deve ser ajustado se o ambiente virtual usar outra
versão de Python. Para tornar a configuração automática ao ativar o ambiente,
adicione esse trecho ao arquivo `.venv/bin/activate`:

```bash
V="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

export LD_LIBRARY_PATH="$(
  find "$V/lib"/python*/site-packages/nvidia \
    -type d -name lib -printf '%p:' |
    sed 's/:$//'
)${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

É sugerido adicionar esse código ao final do arquivo.


Se já estiver com o ambiente ativado enquanto realiza a configuração, apenas reative o ambiente e teste:

```bash
deactivate
source .venv/bin/activate
```

Valide a detecção:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

O resultado esperado contém um dispositivo com tipo `GPU`:

```text
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

### Kernel do VS Code e Jupyter

O kernel do notebook precisa usar o Python do mesmo ambiente virtual e herdar
o `LD_LIBRARY_PATH` configurado. A opção preferencial é abrir o projeto
diretamente no VS Code conectado ao WSL2:

Antes disso, instale no VS Code as extensões **WSL**, **Python** e **Jupyter**.
A extensão WSL permite que o comando `code .` abra o projeto em uma janela
remota Linux, em vez de usar o Python do Windows.

```bash
cd /mnt/d/Projects/26.2/TADI/replicating_RePO
source .venv/bin/activate
code .
```

Na janela remota, identificada como `WSL: Ubuntu`, abra o notebook e selecione
`Python (replicating_RePO - GPU)`. Esse fluxo permite que o VS Code mostre o
kernel local sem copiar a URL de um servidor Jupyter. Se o kernel não aparecer,
execute novamente o comando `ipykernel install` dentro do `.venv` e use
`Developer: Reload Window`.

Como alternativa, é possível iniciar o Jupyter manualmente no WSL2:

```bash
cd /mnt/d/Projects/26.2/TADI/replicating_RePO
source .venv/bin/activate
cd code
jupyter lab
```

Nesse caso, no VS Code use **Select Kernel > Select Existing Jupyter Server**,
cole a URL do servidor local exibida no terminal e escolha
`Python (replicating_RePO - GPU)`. Esse procedimento é uma alternativa quando
o VS Code não estiver conectado ao WSL2 diretamente.

Dentro do notebook, execute a célula de confirmação:

```python
import os
import sys
import tensorflow as tf

print("Python:", sys.executable)
print("Versão:", sys.version)
print("Diretório atual:", os.getcwd())
print("LD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH"))
print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

O Python deve apontar para `.venv/bin/python`, o `LD_LIBRARY_PATH` deve conter
caminhos absolutos dentro de `.venv/lib/python*/site-packages/nvidia` e a lista
de GPUs deve conter `PhysicalDevice(..., device_type='GPU')`.

Quando houver a confirmação de configurações corretas, remova a célula.

### TensorFlow 2.10 no Windows

Também é possível usar TensorFlow 2.10 com CUDA nativo no Windows. Essa
alternativa exige um ambiente separado com versões compatíveis de:

- Python;
- TensorFlow;
- CUDA;
- cuDNN.

Ela não é recomendada para este projeto porque o ambiente nativo atual utiliza
Python 3.13 e TensorFlow 2.21.0. O TensorFlow 2.10 não oferece suporte a
Python 3.13; seria necessário criar um ambiente separado com Python suportado
por essa versão antiga.
Além disso, misturar versões antigas no mesmo ambiente pode quebrar as
dependências dos notebooks.

## Uso do Google Colab

O Google Colab fornece um ambiente Linux remoto e pode disponibilizar uma GPU
NVIDIA mesmo quando o computador local não possui GPU física. O processamento
ocorre nos servidores do Colab; a GPU do computador local não é utilizada.

### Ativar uma GPU

No Colab, selecione:

```text
Runtime → Change runtime type → Hardware accelerator → GPU
```

Em uma interface traduzida, a opção pode aparecer como:

```text
Ambiente de execução → Alterar tipo de ambiente de execução → Acelerador de hardware → GPU
```

A disponibilidade e o modelo da GPU dependem da cota e do tipo de conta. O
Colab pode fornecer, por exemplo, uma NVIDIA T4 ou L4.

Confirme a GPU atribuída:

```python
!nvidia-smi
```

Depois confirme que o TensorFlow a detectou:

```python
import tensorflow as tf

gpus = tf.config.list_physical_devices("GPU")

print("TensorFlow:", tf.__version__)
print("GPUs disponíveis:", gpus)

if not gpus:
    raise RuntimeError(
        "Nenhuma GPU foi detectada. Selecione uma GPU nas configurações "
        "do ambiente de execução do Colab."
    )

for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
```

O `set_memory_growth` faz com que o TensorFlow aloque memória conforme a
necessidade, em vez de reservar toda a memória da GPU no início.

### Baixar e executar o projeto

```python
%cd /content
!git clone https://github.com/marquesgbr/replicating_RePO.git
%cd /content/replicating_RePO/code
```

Os notebooks utilizam caminhos relativos à pasta `code`, portanto é importante
executá-los a partir dessa pasta.

O TensorFlow já costuma estar instalado no Colab. Antes de instalar uma nova
versão, verifique o ambiente:

```python
!python --version
!pip show tensorflow keras
```

Evite executar automaticamente `pip install -r requirements.txt` no Colab,
porque o requisito `tensorflow` não fixa uma versão e pode substituir a versão
pré-configurada do Colab. Essa substituição pode criar incompatibilidades com
CUDA, Keras ou outras bibliotecas do runtime.

Instale somente dependências que estejam faltando. Por exemplo:

```python
%pip install pandas numpy ipykernel
```

Se for necessário instalar uma versão específica do TensorFlow, reinicie o
runtime depois da instalação:

```python
%pip install "tensorflow==2.18.*"
```

Em seguida, use:

```text
Runtime → Restart session
```

### Versões antigas do Python no Colab

O Colab oferece algumas versões anteriores do runtime por meio de:

```text
Runtime → Change runtime type → Runtime Version
```

Essa opção é preferível a instalar manualmente outro Python dentro da sessão,
porque mantém Python, TensorFlow, CUDA e as bibliotecas do sistema compatíveis.
Runtimes antigos podem deixar de estar disponíveis depois de determinado
período.

Não é necessário usar Python antigo apenas para habilitar CUDA no Colab. O
Colab já utiliza Linux e fornece um ambiente preparado para GPU. A utilização
de TensorFlow 2.10 só seria relevante para o caso específico de GPU CUDA
nativa no Windows.

Instalar outro Python manualmente dentro do Colab pode causar problemas no
kernel do Jupyter, nos caminhos do `pip` e nas bibliotecas binárias do
TensorFlow. Só faça isso se houver uma necessidade específica de reproduzir
um ambiente antigo.

## Carregamento dos modelos deste projeto

Os modelos versionados em `models/` estão no formato SavedModel legado do
TensorFlow. Em Keras 3, esta chamada pode falhar:

```python
tf.keras.models.load_model("../models/pkt_model/")
```

Use o adaptador fornecido pelo projeto:

```python
from load_model_compat import load_model

model = load_model("../models/pkt_model/")
```

Para o modelo baseado em fluxos:

```python
model = load_model("../models/flw_model/")
```

O adaptador está em `code/load_model_compat.py`. Ele carrega a assinatura
`serving_default`, converte a saída única para um tensor e preserva os
gradientes necessários pelos notebooks adversariais.

## Verificar se o cálculo está indo para a GPU

Durante o diagnóstico, é possível habilitar o registro de dispositivos:

```python
tf.debugging.set_log_device_placement(True)

x = tf.random.normal((4096, 4096))
y = tf.matmul(x, x)

print(y.device)
```

O dispositivo esperado é semelhante a:

```text
/device:GPU:0
```

Desative os logs depois do teste:

```python
tf.debugging.set_log_device_placement(False)
```

Também é possível acompanhar a utilização enquanto o notebook executa:

```python
!nvidia-smi
```

Leitura de CSV, operações NumPy, preparação de dados e operações pequenas
podem continuar na CPU. Isso é normal. O objetivo é que as operações maiores
do TensorFlow, como as camadas densas e multiplicações matriciais, sejam
executadas na GPU.

## Referências

- [TensorFlow: instalação](https://www.tensorflow.org/install/pip)
- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Google Colab: versões anteriores do runtime](https://research.google.com/colaboratory/runtime-version-faq.html)
