# Fluxo completo: EDF → CNN (limiar duplo) → revisão → BiMamba

Este documento descreve o pipeline atual de preparação de dados de RSWA, do
arquivo EDF bruto até o tensor `.pt` pronto para treinar o modelo BiMamba,
incluindo o novo modo de revisão binária.

## 1. EDF → `.pt` (preprocessamento + rotulagem)

`preprocess_exam` (em `src/sleep_rswa/preprocessing/preprocess.py`) lê um EDF
bruto + hipnograma (`.mat`) e produz um tensor `.pt` com o schema canônico:

- `signals` `[T, 5, 300]` — 3 EEG + EOG + EMG mento, 100 Hz, mini-épocas de 3s.
- `sleep_stages` `[T]` — 0=W, 1=N1, 2=N2, 3=N3, 4=REM, -1=não estagiado.
- `tonic_labels`, `phasic_labels`, `any_labels` `[T]` float — as 3 cabeças de
  rótulo binário por mini-época (`any = tonic OR phasic`, mais tolerante a
  desalinhamento de fronteira).
- `tonic_cov`, `phasic_cov`, `any_cov` `[T]` float — cobertura fracionária de
  cada rótulo dentro da mini-época (usada como peso/confiança, não é 0/1 puro).
- `rswa_conf`, `rswa_labels`, `rem_baseline_uv`, `rem_baseline_n_epochs`.
- `label_source` — string indicando a origem do rótulo: `"csv"` (revisão
  humana em `view/revisado/*_revisado.csv`, rasterizada por
  `rswa_labels.rasterize_rswa_annotations`) ou
  `"auto_cnn_limiar_duplo_v1"` (CNN + limiar duplo, ver §2).

A escolha da fonte é o parâmetro `rswa_source="csv"|"auto"` de
`preprocess_exam` — **não existe um script orquestrador separado**: a lógica
de rotulagem automática está embutida dentro de `preprocess_exam`, entre os
passos numerados 10 (rasterização/CNN) e 11 (baseline REM), por decisão
explícita do usuário.

## 2. Rotulagem automática: CNN + limiar duplo

Quando `rswa_source="auto"`, `preprocess_exam` chama
`auto_label_rswa_from_signals` (`src/sleep_rswa/preprocessing/auto_rswa.py`):

1. A CNN de detecção de movimento (`classifier/movement_clf/model.py`,
   checkpoint padrão `classifier/outputs/movement_cnn_final.pt`) pontua cada
   mini-época do canal EMG, gerando um score de probabilidade de movimento.
2. O limiar duplo (histerese: `K_ON`/`K_OFF` em
   `testes/src/limiar/threshold_rule.py`) transforma os scores em segmentos
   candidatos, confirma/descarta por duração e amplitude mínima, e classifica
   cada evento confirmado como tônico ou fásico pela duração
   (`PHASIC_LO_S`/`PHASIC_HI_S` vs. `TONIC_MIN_DUR_S`).
3. Os eventos confirmados são rasterizados de volta para os 3 vetores
   `tonic_labels`/`phasic_labels`/`any_labels` (+ `*_cov`), no mesmo formato
   que a rotulagem humana produz — o restante do pipeline (treino, revisão)
   não precisa saber qual foi a origem.

Esta lógica foi portada e verificada com paridade numérica exata contra o
script original `classifier/auto_label.py` (mesmas contagens de candidatos,
eventos confirmados, descartados e somas por tipo, em exames reais).

`classifier/apply_labels.py` é o passo **humano pós-CNN**: quando existe um
CSV revisado (`view/revisado/<exame>_revisado.csv`) para um exame que foi
rotulado automaticamente, ele sobrescreve `tonic_labels`/`phasic_labels` (e
recalcula `any_labels`/`any_cov` com a mesma semântica de precedência) direto
no `.pt`, preservando `label_source` como rastro de auditoria.

## 3. Revisão humana

Duas interfaces web, servidas por `view/app.py`:

- **Modo revisor** (`--mode revisor`, padrão): fluxo original. Roda a CNN ao
  vivo sobre o EMG, permite editar tipo (tônico/fásico), apagar falsos
  positivos e adicionar eventos manuais; salva em
  `view/revisado/<exame>_revisado.csv`.
- **Modo revisão** (`--mode revisao`, novo): **não roda a CNN**. Lê
  diretamente as 3 cabeças de rótulo já gravadas no `.pt`
  (`tonic_labels`/`phasic_labels`/`any_labels`, de qualquer origem — CSV
  humano ou CNN+limiar-duplo) e apresenta cada evento (run contíguo de rótulo
  positivo) para uma decisão binária: **correto** ou **incorreto**, com nota
  opcional. Serve para auditar rapidamente a qualidade da rotulagem já
  aplicada a um exame, sem re-editar onset/duração/tipo.

  Decisões são gravadas por exame em
  `view/revisao_binaria/<exame>_revisao.csv` (upsert por `event_id`).
  `POST /api/revisao/report` (ou `GET /api/revisao/report?names=...`) agrega
  as decisões de uma lista de exames num relatório de acurácia por tipo
  (tonic/phasic/any + total), salvo em
  `view/revisao_binaria/relatorio_revisao_binaria.csv` e `.md`.

  Rodar: `python3 view/app.py --mode revisao --data classifier/data --port 8000`.
  A UI é `view/revisao.html` (página separada de `view/index.html`, mesmo
  processo/servidor).

## 4. `.pt` → BiMamba

Os `.pt` com `label_source` confiável (humano, ou CNN+limiar-duplo já
auditado pelo modo revisão) alimentam o treino do modelo BiMamba para
estagiamento do sono (W/N1/N2/N3/REM) e detecção de RSWA por mini-época de 3s,
conforme a proposta do projeto.

## Arquivos-chave

| Arquivo | Papel |
|---|---|
| `src/sleep_rswa/preprocessing/preprocess.py` | `preprocess_exam`: EDF→`.pt`, chama rasterização humana ou `auto_rswa` |
| `src/sleep_rswa/preprocessing/rswa_labels.py` | Rasteriza CSV humano → `tonic/phasic/any_labels` + `*_cov` |
| `src/sleep_rswa/preprocessing/auto_rswa.py` | CNN + limiar duplo, embutido em `preprocess_exam` (passos 10→11) |
| `classifier/auto_label.py` | Script original standalone (referência/paridade), roda no `.pt` já salvo |
| `classifier/apply_labels.py` | Aplica CSV humano revisado sobre `.pt` já rotulado automaticamente (`any`) |
| `view/app.py` | Servidor dos dois modos (`revisor` e `revisao`) |
| `view/index.html` | UI do modo revisor (edição completa) |
| `view/revisao.html` | UI do modo revisão (binário correto/incorreto) |
| `view/revisado/*.csv` | Saída do modo revisor (rótulos editados) |
| `view/revisao_binaria/*.csv`, `relatorio_revisao_binaria.{csv,md}` | Saída do modo revisão (decisões + relatório) |
