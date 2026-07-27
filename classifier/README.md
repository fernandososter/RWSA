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


# PASSOS PARA TREINAR E EXECUTAR O MODELO: 

# 1 - ADICIONAR LABELS AO .PT
python classifier/apply_labels.py --dry-run     # confere antes, nao grava nada
python classifier/apply_labels.py               # aplica de fato, com backup automatico

# 2 - TRAIN LOSO
python classifier/train_loso.py

# 3 - EVALUATE
python classifier/evaluate_loso.py

# 4 - TREINAMENTO FINAL
python classifier/train_final.py


# 5 - INFERENCIA: 
python classifier/predict_movements.py caminho/EXAME.pt -o EXAME_movimentos.csv