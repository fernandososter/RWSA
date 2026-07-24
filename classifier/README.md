# Movement Classifier (isolado)

Detector binário de **movimento** (atividade EMG, a noite toda) treinado nos 4 exames anotados.

**Este módulo é totalmente independente** de `src/sleep_rswa`. Não importa nada do
projeto original. O único ponto de contato é o *formato* dos arquivos:

- **Entrada**: arquivos `.pt` (dict com `signals [T,5,300]`, `sleep_stages [T]`,
  e, para treino, `tonic_labels`/`phasic_labels [T]`). Canal 4 = EMG do mento.
- **Saída**: CSV de anotações `subject_id, onset_s, duration_s, type, score`
  (mesmo formato dos `*_rswa.csv`), com `type=movement`.

## Estrutura
```
classifier/
  movement_clf/       código do módulo (dataset, cnn, treino, avaliação, inferência)
  data/               os .pt de treino (rbd1, rbd2, rbd3, rbd5)
  outputs/            métricas, figuras, checkpoints, CSVs gerados
  train_loso.py       validação leave-one-subject-out
  train_final.py      treino final nos 4 exames
  predict_movements.py    inferência: novo .pt -> CSV de movimentos
```
