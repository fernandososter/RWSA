# Planejamento: basal de REM no pipeline .pt (EDF -> .pt)

Sem alteracao de codigo nesta etapa — apenas levantamento, estimativa de esforco e
discussao de opcoes de projeto.

## 1. Estado atual da pipeline (verificado no codigo, nao por memoria)

Entrada: `scripts/parser.py` -> `run_preprocessing_parallel(edf_dir)` (ou serial
`run_preprocessing`) -> `preprocess_exam()` por exame -> `_save_result()` grava o `.pt`.
Todo o pacote `src/sleep_rswa/preprocessing/` e AUTOCONTIDO: nao importa nada de
`src/sleep_rswa/config.py` nem de `data.py` (confirmado por grep).

Etapas de `preprocess_exam` (docstring do modulo, `preprocess.py`):
1. Carrega EDF + hipnograma (.mat) + CSV de RSWA (se houver)
2. Resolve canais (ausentes -> zeros + channel_mask=False)
3. Constroi stage_map, cropa o raw em [annot_start, annot_end]
4. Filtra por tipo (EMG 10-40Hz / EEG / EOG) + notch 60Hz, reamostra p/ 100Hz
5. Epocas de 30s -> stages por epoca
6. Zero-fill dos canais ausentes -> matriz (n_epochs, 5, 3000)
7. Sub-segmenta 30s -> mini-epocas de 3s (300 amostras)
8. Expande stages 30s -> mini-epocas
9. Rasteriza eventos RSWA -> tonic_labels/phasic_labels/rswa_labels/rswa_conf por mini-epoca

Formato salvo hoje (torch.save, em `_save_result`):
```
signals        Tensor (T, 5, 300)  float32   <- BRUTO, sem normalizacao nenhuma
sleep_stages   Tensor (T,)         int64
channel_mask   Tensor (5,)         bool
channel_names  list[str|None]
tonic_labels   Tensor (T,)         float32 {0,1}
phasic_labels  Tensor (T,)         float32 {0,1}
rswa_labels    Tensor (T,)         int64   {0,1,2,3}
rswa_conf      Tensor (T,)         float32 {0,1}
```
Canal EMG = indice 4 (`RSWAConfig.emg_channel_index`, definido em `src/sleep_rswa/config.py`
e espelhado em `classifier/movement_clf/dataio.py::EMG_CHANNEL_INDEX`).

Confirmado por leitura direta de um `.pt` real (`classifier/data/rbd1.pt`):
`signals[:,4,:]` esta em Volts (std ~4.5e-6 V = 4.5 uV), condizente com EMG mento
pos-filtro. NENHUM z-score esta presente no arquivo salvo.

## 2. Onde a normalizacao acontece hoje (fora do .pt, em tempo de carregamento)

Duas implementacoes INDEPENDENTES e propositalmente desacopladas:

- `src/sleep_rswa/data.py::_zscore_per_channel` — usado por `SleepAnalysisDataset`
  (RSWADetectionNet / staging). Z-score por canal com media/desvio da NOITE INTEIRA
  do exame (todas as mini-epocas, nao so REM).
- `classifier/movement_clf/dataio.py::zscore_emg` — usado pelo classificador de
  movimento isolado. Mesma logica, so do canal EMG, tambem noite inteira. Este modulo
  explicitamente NAO importa nada de `src/sleep_rswa` (isolamento documentado no
  docstring do arquivo).

Terceira abordagem, ja validada nesta sessao nos testes deterministicos
(`testes/src/limiar`, `tkeo`, `cusum_glr`): baseline LOCAL por janela deslizante
(120s, percentil 10) sobre o envelope RMS do EMG BRUTO (sem z-score), com
`score = pico/baseline >= 2.0` como criterio de classificacao tonico/any/fasico.
BASELINE_WIN_S=120, BASELINE_PCT=10 (`threshold_rule.py`).

## 3. Discussao: basal antes ou depois da normalizacao?

Duas perguntas ortogonais.

### 3a. Unidade: microvolts brutos vs. z-score

| Criterio | Bruto (uV), antes de normalizar | Normalizado (z-score), depois |
|---|---|---|
| Interpretabilidade clinica | Direta, auditavel sem contexto extra | So faz sentido junto com a formula de normalizacao usada |
| Compatibilidade com as 2 pipelines existentes | Uma formula so, consumida por ambas | Obriga a escolher UMA das duas formulas (`data.py` ou `dataio.py`); a outra pipeline recalcularia do zero ou importaria valor de convencao alheia, quebrando o isolamento intencional |
| Robustez a artefato de vigilia entre exames | Nao afetado | Estatistica "noite inteira" muda com quantidade de artefato/movimento em vigilia — dois exames com REM fisiologicamente identico podem receber basais diferentes so por isso |
| Efeito na deteccao por RAZAO (pico/basal) | Se a mesma transformacao for aplicada a pico e basal, a razao e invariante a escala — calcular antes ou depois nao muda a classificacao, desde que consistente | Mesmo argumento, mas o valor ja fixado obriga o consumidor a saber qual formula foi usada |

RECOMENDACAO: gravar em microvolts BRUTOS, calculado direto do canal EMG ja salvo em
`signals` (hoje nada ali passa por normalizacao — "antes" e simplesmente "como esta").
Quem precisar de versao normalizada divide pelo mesmo desvio-padrao que ja calcula
para normalizar o sinal, no ponto de consumo.

### 3b. Estatistica: media simples vs. percentil vs. excluindo eventos

- Media de TODO o REM (conforme descrito pelo usuario): simples, mas mini-epocas
  com burst tonico sustentado entram na media e puxam o basal para cima —
  um evento real proximo de 1.5x esse basal inflado pode nao ser detectado.
- Media do REM EXCLUINDO mini-epocas ja rotuladas tonico/fasico: mais fiel ao
  conceito clinico de "tono de fundo" (AASM/SINBAR), mas exige rotulos ja
  existentes — funciona para os 58 exames revisados, mas nao ha rotulo em
  exames novos na hora da inferencia.
- Percentil baixo (ex. 10 = mesmo BASELINE_PCT ja validado nos testes
  deterministicos) do envelope REM, SEM excluir nada: robusto a contaminacao
  por eventos sem precisar de rotulo, e funciona igual em exames rotulados
  (treino) e nao rotulados (inferencia).

RECOMENDACAO: percentil (mesma formula usavel em treino e inferencia).

## 4. Plano de esforco (pipeline de geracao do .pt)

| # | Tarefa | Onde | Esforco |
|---|---|---|---|
| 1 | Especificar funcao (nome do campo, assinatura, tratamento de exame sem REM) | design | 0.5-1h |
| 2 | Implementar `compute_rem_baseline()` — novo modulo `rem_baseline.py`, espelhando o padrao de `rswa_labels.py` | `src/sleep_rswa/preprocessing/` | 1-2h |
| 3 | Integrar no `preprocess_exam()`, logo apos a rasterizacao RSWA (etapa 9) | `preprocess.py` | 0.5h |
| 4 | Persistir o novo campo no `.pt` | `preprocess.py::_save_result` | 0.25h |
| 5 | Atualizar docstring do modulo (secao "Formato salvo") | `preprocess.py` | 0.25h |
| 6 | Script de backfill para os 60 .pt ja existentes — NAO precisa reprocessar EDFs, ja que `signals` bruto + `sleep_stages` ja estao salvos; le, calcula, regrava preservando as demais chaves | novo script | 1-2h |
| 7 | QC do backfill: CSV com o basal por sujeito p/ inspecao de outliers, diff conferindo que as demais chaves ficaram identicas | mesmo script | incluso no item 6 |
| 8 | Teste end-to-end a partir de 1-2 EDFs brutos (nao so backfill), p/ confirmar que a pipeline completa tambem grava o campo corretamente | validacao | 0.5-1h |
| 9 | Testes unitarios (estender `tests/test_shapes.py`): presenca/dtype/faixa plausivel do novo campo | testes | 0.5-1h |
| 10 | Documentacao (README/RELATORIO ou docstring) do novo campo e consumo esperado | docs | 0.5h |

TOTAL ESTIMADO: ~6-10 horas (~1 dia de trabalho focado). Mudanca pequena e de
baixo risco: um unico campo novo, opcional, compativel com o padrao `.get()` ja
usado em todo o loader (`data.py`, `dataio.py` ja toleram chaves ausentes).

## 5. Fatos de dados relevantes (checados nos 60 .pt existentes)

- REM por sujeito varia de 12.5 a 201 minutos (media ~82 min); TODOS os 60
  sujeitos tem pelo menos 250 mini-epocas de REM (>= 12.5 min) — nenhum caso de
  REM zero ou quase-zero a tratar como excecao no calculo do basal hoje.
- 58 dos 60 .pt em `classifier/data/` tem CSV revisado correspondente
  (`_revisado.csv`) com tonic_labels/phasic_labels ja gravados; os 2 restantes
  (ADRYAN..., ALAN...) tem .pt mas ainda sem revisao/rotulo.

## 6. Decisao pendente antes de implementar

Confirmar: (a) unidade = microvolts brutos (nao normalizado); (b) estatistica =
percentil baixo (ex. 10) do envelope REM, sem exclusao de eventos. Com isso
definido, a implementacao segue direto pelo plano de esforco acima.
