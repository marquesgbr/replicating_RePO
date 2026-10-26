# Checklist do projeto

Use este arquivo para acompanhar o andamento da reprodução dos experimentos e
da escrita do relatório. Marque uma tarefa somente quando os critérios de
conclusão estiverem atendidos e registre evidências, resultados ou observações
nas linhas correspondentes.

## 1. Organização e preparação

- [x] Organizar a documentação em `docs/`.
  - [x] Documentar compatibilidade do TensorFlow com GPU.
  - [x] Documentar execução no Colab pelo VS Code.
  - [x] Documentar execução no Colab pelo navegador.
  - [x] Documentar os requisitos do relatório.
- [ ] Definir o ambiente oficial de execução do grupo.
  - [ ] Escolher entre Linux/WSL2 com GPU, CPU local ou Colab.
  - [ ] Registrar Python, TensorFlow, Keras, CUDA/cuDNN, sistema operacional e
        kernel utilizados.
  - [ ] Confirmar que todos os membros conseguem abrir os notebooks no
        ambiente escolhido.
- [ ] Validar o ambiente.
  - [ ] Confirmar as versões instaladas.
  - [ ] Confirmar a disponibilidade da GPU, quando aplicável.
  - [ ] Confirmar que `load_model_compat` pode ser importado.
  - [ ] Confirmar que os modelos de fluxo e pacotes podem ser carregados.
  - [ ] Registrar problemas de instalação e soluções adotadas.

## 2. Dados

- [ ] Confirmar a origem e a integridade dos arquivos de dados.
  - [ ] Confirmar os arquivos compactados necessários.
  - [ ] Confirmar os metadados de normalização em `data/*/x_train_meta/`.
  - [ ] Verificar se os arquivos grandes não serão adicionados ao Git.
- [ ] Executar `extract_flow_data.sh`.
- [ ] Executar `extract_packet_data.sh`.
- [ ] Verificar as pastas extraídas de fluxo e pacotes.
- [ ] Verificar quantidades, formatos, shapes, rótulos e valores ausentes.
- [ ] Registrar a divisão entre treino, validação e teste, quando aplicável.
- [ ] Registrar o commit e o ambiente usados na preparação dos dados.

## 3. Reprodução dos experimentos de referência

Execute os notebooks normais antes dos adversariais.

### 3.1 Modelo baseado em fluxos

- [ ] Executar `code/Flow_based_RePO_Normal_Test.ipynb`.
- [ ] Salvar os resultados relevantes do teste normal.
- [ ] Executar `code/Flow_based_RePO_Adversarial.ipynb`.
- [ ] Salvar os resultados relevantes do teste adversarial.

### 3.2 Modelo baseado em pacotes

- [ ] Executar `code/Packet_based_RePO_Normal_Test.ipynb`.
- [ ] Salvar os resultados relevantes do teste normal.
- [ ] Executar `code/Packet_based_RePO_Adversarial.ipynb`.
- [ ] Salvar os resultados relevantes do teste adversarial.

### 3.3 Registro de reprodutibilidade

- [ ] Registrar o notebook executado.
- [ ] Registrar o kernel e o interpretador Python.
- [ ] Registrar as versões das bibliotecas.
- [ ] Registrar o sistema operacional e a GPU, quando aplicável.
- [ ] Registrar a data e o commit do repositório.
- [ ] Preservar ou exportar as saídas necessárias para o relatório.

## 4. Análise dos resultados de referência

- [ ] Consolidar as métricas dos testes normais.
- [ ] Consolidar as métricas dos testes adversariais.
- [ ] Verificar os limiares utilizados pelos notebooks.
- [ ] Gerar tabelas comparando fluxo e pacotes.
- [ ] Gerar gráficos necessários para o relatório.
- [ ] Explicar diferenças entre os resultados reproduzidos e os resultados
      reportados no artigo de referência.
- [ ] Registrar limitações, variações e falhas observadas durante a reprodução.

## 5. Dataset adicional

Esta seção deve ser concluída caso o escopo do projeto inclua outro conjunto de
dados além dos dados usados pelo artigo de referência.

- [ ] Escolher o dataset adicional.
- [ ] Justificar a escolha.
- [ ] Documentar origem, características e classes.
- [ ] Preparar os dados.
- [ ] Definir treino, validação e teste.
- [ ] Registrar quantidades por classe.
- [ ] Executar a avaliação da solução de referência no dataset adicional.
- [ ] Consolidar os resultados para comparação.

## 6. Melhoria proposta pela equipe

Esta seção deve ser concluída caso o grupo implemente uma melhoria em relação à
solução do artigo de referência.

- [ ] Definir o objetivo da melhoria.
- [ ] Descrever a arquitetura proposta.
- [ ] Descrever entradas e saídas dos componentes.
- [ ] Descrever os algoritmos e métodos empregados.
- [ ] Justificar as escolhas de projeto.
- [ ] Implementar a melhoria.
- [ ] Criar ou atualizar testes necessários.
- [ ] Avaliar a melhoria nos dados do artigo de referência.
- [ ] Avaliar a melhoria no dataset adicional, se aplicável.
- [ ] Comparar a melhoria com a solução de referência.

## 7. Relatório

Os requisitos detalhados estão em
[requisitos_relatório.md](./requisitos_relatório.md).

### Abstract/Resumo

- [ ] Contextualizar o problema.
- [ ] Apresentar a solução.
- [ ] Resumir a metodologia.
- [ ] Apresentar os principais resultados.

### 1. Introdução

- [ ] Explicar o problema investigado.
- [ ] Justificar a relevância do problema.
- [ ] Explicar a motivação do artigo de referência.
- [ ] Apresentar soluções existentes e limitações.
- [ ] Apresentar a solução proposta.
- [ ] Listar as principais contribuições.

### 2. Trabalhos relacionados

- [ ] Selecionar trabalhos relacionados.
- [ ] Descrever contribuições e limitações.
- [ ] Comparar os trabalhos com o artigo de referência.

### 3. Modelo de ameaça

- [ ] Descrever os ataques considerados.
- [ ] Documentar as premissas dos ataques.
- [ ] Incluir figuras, algoritmos ou equações quando necessário.

### 4. Sistema proposto pelo artigo de referência

- [ ] Descrever a arquitetura.
- [ ] Descrever componentes, entradas e saídas.
- [ ] Explicar o funcionamento do sistema.
- [ ] Explicar os métodos e algoritmos utilizados.

### 5. Solução proposta pela equipe

- [ ] Descrever o que será melhorado.
- [ ] Descrever a arquitetura da melhoria.
- [ ] Descrever componentes, entradas e saídas.
- [ ] Explicar métodos e algoritmos.
- [ ] Justificar as escolhas realizadas.

### 6. Metodologia

- [ ] Descrever os dados do artigo de referência.
- [ ] Descrever treino, validação e teste.
- [ ] Informar quantidades por classe.
- [ ] Descrever o dataset adicional, se houver.
- [ ] Descrever métricas e experimentos.
- [ ] Registrar ambientes e condições de execução.

### 7. Resultados e discussões

- [ ] Apresentar resultados da solução de referência.
- [ ] Apresentar resultados no dataset adicional, se houver.
- [ ] Apresentar resultados da melhoria da equipe, se houver.
- [ ] Comparar resultados com outros trabalhos.
- [ ] Incluir tabelas e gráficos.
- [ ] Interpretar os resultados, sem apenas descrevê-los.

### 8. Conclusão e trabalhos futuros

- [ ] Resumir os principais resultados.
- [ ] Descrever limitações.
- [ ] Discutir problemas da solução de referência.
- [ ] Discutir problemas da melhoria, se houver.
- [ ] Propor trabalhos futuros.

### 9. Referências

- [ ] Revisar todas as citações.
- [ ] Ajustar as referências ao formato IEEE.
- [ ] Confirmar que toda referência citada aparece na bibliografia.

## 8. Revisão e entrega

- [ ] Revisar código, notebooks e documentação.
- [ ] Remover saídas, caches e arquivos temporários que não devam ser
      versionados.
- [ ] Confirmar que datasets grandes não estão no commit.
- [ ] Verificar links relativos da documentação.
- [ ] Executar `git diff --check`.
- [ ] Revisar o histórico e a ordem dos commits.
- [ ] Confirmar que o README descreve o procedimento atual.
- [ ] Confirmar que o relatório compila.
- [ ] Fazer revisão final por outro membro do grupo.
- [ ] Criar o commit final.
- [ ] Fazer push para o repositório remoto.

## Registro de execução (Opcional)

| Data | Pessoa | Ambiente/kernel | Notebook ou tarefa | Commit | Observações |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

