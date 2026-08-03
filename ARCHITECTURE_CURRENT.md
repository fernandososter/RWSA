# Arquitetura Atual do Projeto

Estado documentado em 2026-08-03, refletindo o código presente no working tree atual da branch `dev`, incluindo alterações locais ainda não commitadas.

## 1. Objetivo do repositório

Este projeto hoje combina duas linhas de trabalho relacionadas, mas separadas:

1. `src/sleep_rswa/`
   Pipeline principal de **sleep staging + RSWA** em mini-épocas de 3 segundos.
2. `classifier/` + `view/`
   Pipeline auxiliar de **pré-anotação de movimento EMG** e **revisão humana** para gerar ou corrigir rótulos `tônico` / `fásico`.

O ponto de contato entre essas duas linhas é o **formato dos arquivos `.pt`** e o **formato CSV de anotações**.

## 2. Separação de responsabilidades

### `src/sleep_rswa/`

É o núcleo do projeto principal.

- Faz preprocessamento de EDF para tensores PyTorch.
- Treina modelos de staging e RSWA.
- Carrega EMG como parte do dataset principal.
- Usa rótulos RSWA por mini-época (`tonic_labels`, `phasic_labels`, `rswa_labels`).

Importante:
- O pipeline principal **não depende** de `view/`.
- O pipeline principal **não usa envelope RMS** como entrada do modelo RSWA atual.
- O EMG consumido pelo modelo principal é o **EMG bruto z-scored**.

### `classifier/`

É um módulo **isolado** para detecção binária de movimento EMG na noite toda.

- Não importa nada de `src/sleep_rswa`.
- Usa o mesmo `.pt` como entrada.
- Detecta `movement` por mini-época e funde eventos adjacentes.
- Mais recentemente ganhou uma camada determinística de sub-classificação `fásico` / `candidato a tônico`.

### `view/`

É a interface local de revisão humana.

- Lê os exames `.pt`.
- Pode rodar o detector de movimento diretamente.
- Permite abrir arquivos de anotação já existentes.
- Permite revisar, editar, apagar e salvar eventos finais em CSV.

## 3. Contrato de dados atual

### `.pt` por exame

O formato esperado hoje, na prática, é um `dict` contendo pelo menos:

```python
{
    "subject_id": str,
    "signals": Tensor[T, 5, 300],
    "sleep_stages": Tensor[T],
    "tonic_labels": Tensor[T],   # opcional em exames não anotados
    "phasic_labels": Tensor[T],  # opcional em exames não anotados
    "rswa_labels": Tensor[T],    # no pipeline principal
    "rswa_conf": Tensor[T],      # no pipeline principal
}
```

Convenção importante:

- `signals[T, 5, 300]`
- canal `4` = **EMG do mento**
- `300` amostras por mini-época de 3 s a 100 Hz

## 4. Fluxo ponta a ponta atual

### Fluxo A: preprocessamento principal

1. Um EDF é lido em `src/sleep_rswa/preprocessing/preprocess.py`.
2. O hipnograma `.mat` é alinhado ao registro.
3. O exame é cortado a partir de `annot_start`.
4. Os sinais são filtrados, reamostrados para 100 Hz e segmentados.
5. O resultado é salvo como `.pt` com mini-épocas de 3 s.
6. Se houver CSV de RSWA, ele é rasterizado em `tonic_labels` e `phasic_labels`.

Resultado:
- o `.pt` vira o artefato central compartilhado entre o pipeline principal, o classificador isolado e a interface de revisão.

### Fluxo B: treino/inferência principal de RSWA

1. `src/sleep_rswa/data.py` carrega o `.pt`.
2. O EMG é extraído do canal 4 ou de `emg_signals`, se existir separado.
3. O EMG é **z-scored por canal**.
4. `src/sleep_rswa/models/rswa.py` consome esse EMG bruto normalizado.

Ponto importante:
- o modelo principal atual trabalha sobre **EMG bruto**.
- não há etapa de `envelope RMS` dentro desse modelo.

### Fluxo C: pré-anotação de movimento

1. `classifier/predict_movements.py` carrega o mesmo `.pt`.
2. Extrai o EMG do mento.
3. Aplica z-score por exame.
4. Roda a `MovementCNN`.
5. Produz score por mini-época.
6. Aplica limiar e funde mini-épocas adjacentes em eventos `movement`.
7. Salva um CSV `*_movimentos.csv`.

Esse detector é propositalmente de alta cobertura e baixa precisão, para triagem humana posterior.

### Fluxo D: sub-classificação por envelope RMS

Esse é o principal acréscimo recente.

Arquivo central:
- `classifier/movement_clf/tonic_phasic.py`

Ideia:

1. O detector CNN continua sendo o **localizador grosseiro** de regiões com movimento.
2. Dentro do EMG bruto contínuo, calcula-se um **envelope RMS**.
3. Sobre esse envelope, aplica-se uma regra determinística de amplitude + duração.
4. O resultado é filtrado para manter apenas eventos que se sobrepõem ao movimento detectado pela CNN.

Parâmetros atuais da regra:

- `rms_envelope`: janela de `0.1 s`
- baseline local: percentil `10`
- janela de baseline: `120 s`
- ativação: `envelope >= 2.5 x baseline`
- fusão de gaps: até `1.0 s`
- `phasic`: duração entre `0.5 s` e `5.0 s`
- `tonic_candidate`: duração `>= 16.0 s`

Semântica importante:

- `phasic` é considerado **rótulo final determinístico**.
- `tonic_candidate` **não é rótulo final**.
- `tonic_candidate` significa apenas: "trecho sustentado compatível com tônico, mas que precisa revisão humana".

Justificativa atual do projeto:

- apenas duração + amplitude do EMG não separam de forma confiável tônus tônico verdadeiro de artefato sustentado de movimento;
- por isso a arquitetura atual não transforma automaticamente esses casos em `tonic`.

### Fluxo E: revisão humana em `view/`

1. A app abre um exame `.pt`.
2. Pode:
   - partir das sugestões do `.pt` + detector CNN;
   - abrir um CSV já revisado;
   - abrir um CSV de movimentos;
   - abrir um CSV `*_tonico_fasico.csv`.
3. A revisão acontece sobre o traçado de EMG.
4. O usuário decide `tonic`, `phasic` ou `deleted`.
5. O resultado final é salvo em `view/revisado/<exame>_revisado.csv`.

## 5. Onde o envelope RMS entra hoje

O uso de envelope RMS está **restrito** ao classificador isolado de sub-classificação.

Ele **não** entra:

- no preprocessamento principal;
- no `SleepAnalysisDataset`;
- no `RSWADetectionNet`;
- no detector CNN de movimento como feature de entrada;
- no formato `.pt` salvo.

Ele entra apenas aqui:

1. `classifier/movement_clf/tonic_phasic.py`
   - cálculo do envelope RMS do EMG bruto contínuo
2. `classifier/predict_movements.py`
   - geração opcional do CSV `_tonico_fasico.csv`
3. `view/`
   - leitura desse CSV e suporte à revisão de `tonic_candidate`

Resumo curto:

- **EMG bruto z-scored** continua sendo o sinal principal do sistema.
- **Envelope RMS** é hoje uma camada auxiliar de pós-processamento e apoio à revisão.

## 6. Estado atual da `view/`

A `view/` deixou de ser apenas um revisor simples de eventos detectados pelo modelo.
Hoje ela funciona como um hub local de curadoria.

Capacidades atuais:

- rodar o detector sobre um `.pt`;
- navegar evento a evento;
- visualizar a noite inteira;
- criar eventos manuais;
- excluir mini-janelas dentro de um evento fundido;
- salvar CSV revisado;
- configurar `meas_date` do EDF para converter `onset_s` do tempo do `.pt` para o tempo do EDF;
- abrir arquivos de anotações de múltiplas origens.

Arquivos de anotação reconhecidos:

- `*_revisado.csv`
  - decisões finais confirmadas por humano
- `*_movimentos.csv`
  - saída do detector binário de movimento
- `*_tonico_fasico.csv`
  - saída da regra RMS com `phasic` e `tonic_candidate`

Também houve mudança recente no frontend para:

- exibir `tonic_candidate`;
- marcar eventos com `needs_review`;
- filtrar a lista por tipo e necessidade de revisão;
- avisar antes de salvar se ainda houver candidatos a tônico sem decisão final.

## 7. Conversão temporal

O projeto hoje trabalha com dois referenciais de tempo:

### Tempo do `.pt`

- origem = mini-época 0 do tensor
- usado internamente pelo detector e por boa parte da revisão

### Tempo do EDF

- origem = início real da gravação EDF
- necessário para compatibilidade com os CSVs clínicos originais e para reuso no preprocessamento

Conversão atual:

```text
onset_edf = annot_start + onset_pt
annot_start = hipno_start - meas_date
```

Com correção de meia-noite quando necessário.

Na arquitetura atual:

- `view/` salva `meas_date` por exame em `view/exam_config.json`
- `view/mat/hyp_<exame>.mat` fornece o início do hipnograma
- `classifier/predict_movements.py` também aceita `--meas-date`, `--hipno-start` ou `--annot-start`

## 8. Mudanças recentes mais importantes

### Já presentes no histórico recente

- crescimento grande do conjunto de CSVs revisados em `view/revisado/` e `classifier/labels/`
- consolidação do fluxo de `meas_date` / `annot_start`
- amadurecimento do revisor `view/`

### Ainda locais no working tree atual

- novo arquivo `classifier/movement_clf/tonic_phasic.py`
- `classifier/predict_movements.py` agora gera um segundo CSV `_tonico_fasico.csv`
- `view/app.py` ganhou listagem e abertura de múltiplos CSVs de anotação
- `view/index.html` ganhou suporte a `tonic_candidate`, `needs_review` e filtros

## 9. Decisões arquiteturais implícitas do estado atual

### 1. Manter isolamento entre pipelines

`classifier/` continua isolado de `src/sleep_rswa/`.

Isso reduz acoplamento, mas implica duplicação de algumas lógicas simples, como:

- parsing de `HH:MM:SS`
- resolução de `annot_start`
- contrato dos `.pt`

### 2. Usar revisão humana como parte central do processo

O sistema atual não tenta fechar o problema de tônico/fásico de ponta a ponta só com modelo.

Em vez disso:

- a CNN encontra regiões candidatas;
- a regra RMS ajuda a priorizar e separar casos fáceis;
- a interface humana continua sendo a fonte final da verdade.

### 3. Tratar `tonic_candidate` como estado intermediário, não como classe final

Essa é provavelmente a decisão mais importante do estado atual.

Hoje a arquitetura assume:

- `phasic` pode ser promovido automaticamente pela regra determinística;
- `tonic` não deve ser promovido automaticamente a partir apenas de duração + envelope RMS.

## 10. Riscos e pontos de atenção

### Acoplamento por contrato de arquivo

Embora os módulos sejam isolados por import, eles dependem fortemente de:

- formato do `.pt`
- nomes e semântica das colunas do CSV
- convenção temporal (`pt` vs `EDF`)

Mudanças nesses contratos têm impacto transversal.

### Duplicação de lógica temporal

`view/` e `classifier/predict_movements.py` mantêm lógica semelhante para offsets e horários.
Se uma regra mudar, existe risco de divergência.

### RMS ainda é pós-processamento, não treinamento

O uso de envelope RMS ainda não está integrado ao dataset ou ao modelo principal.
Então hoje ele ajuda na revisão, mas não necessariamente realimenta o modelo com a mesma representação.

## 11. Resumo executivo

O estado atual da arquitetura pode ser resumido assim:

- o artefato central do projeto é o `.pt` por exame;
- o pipeline principal (`src/`) usa **EMG bruto z-scored** para staging + RSWA;
- o módulo isolado (`classifier/`) usa outra CNN para detectar **movimento** na noite toda;
- uma mudança recente adicionou uma regra por **envelope RMS** para sub-classificar eventos detectados como `phasic` ou `tonic_candidate`;
- `tonic_candidate` não é rótulo final e exige revisão humana;
- a `view/` virou a camada de curadoria que integra detector, sub-classificação e exportação final para CSV revisado.

Em outras palavras:

- **o EMG bruto continua sendo a base do sistema**
- **o envelope RMS hoje é uma heurística auxiliar de pós-processamento e revisão**
- **a verdade final ainda é produzida pela revisão humana**
