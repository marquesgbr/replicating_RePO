# Execução no Google Colab pelo navegador

Este documento descreve o fluxo alternativo para executar o projeto diretamente
na interface web do Google Colab. Para trabalhar pelo VS Code, consulte a seção
de Colab no [README.md](./README.md) e o documento de compatibilidade em
[gpu_tensorflow_compatibility.md](./gpu_tensorflow_compatibility.md).

## 1. Criar um runtime com GPU

No navegador, abra:

```text
Runtime → Change runtime type → Hardware accelerator → GPU
```

Em português, o menu pode aparecer como:

```text
Ambiente de execução → Alterar tipo de ambiente de execução
  → Acelerador de hardware → GPU
```

Depois confirme a GPU:

```python
!nvidia-smi
```

A GPU é executada nos servidores do Colab; a máquina local não precisa ter uma
GPU física. A disponibilidade do tipo de GPU depende da cota e do runtime
disponível.

## 2. Clonar o repositório

Execute em uma célula:

```python
%cd /content
!git clone https://github.com/marquesgbr/replicating_RePO.git
%cd /content/replicating_RePO/code
```

Se o repositório já existir na sessão:

```python
%cd /content/replicating_RePO/code
```

## 3. Preparar o ambiente

Abra [code/Colab_Setup.ipynb](./code/Colab_Setup.ipynb) no Colab e execute as
células em ordem. O roteiro verifica a GPU, configura a pasta `code`, importa
`load_model_compat` e testa os modelos salvos.

O TensorFlow pré-instalado deve ser mantido sempre que possível. Evite:

```python
%pip install -r requirements.txt
```

O arquivo contém o requisito genérico `tensorflow`, que pode substituir a
versão preparada pelo Colab e causar incompatibilidade com CUDA ou Keras.

## 4. Verificar o TensorFlow

```python
import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))
```

O resultado esperado contém um dispositivo do tipo `GPU`.

## 5. Executar os notebooks

Execute nesta ordem:

1. `Flow_based_RePO_Normal_Test.ipynb`;
2. `Packet_based_RePO_Normal_Test.ipynb`;
3. `Flow_based_RePO_Adversarial.ipynb`;
4. `Packet_based_RePO_Adversarial.ipynb`.

Os dados extraídos precisam existir em `data/`. A sessão do Colab é
temporária; arquivos criados durante a execução podem ser perdidos quando o
runtime for encerrado. Salve resultados importantes no Google Drive ou baixe
os arquivos antes de desconectar.

## 6. Reinício do runtime

Se for necessário instalar uma versão diferente do TensorFlow, reinicie o
runtime depois da instalação:

```text
Runtime → Restart session
```

Depois do reinício, execute novamente o roteiro
[Colab_Setup.ipynb](./code/Colab_Setup.ipynb) desde o início.
