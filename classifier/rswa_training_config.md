# Configuração recomendada para treino do RSWADetectionNet (3 cabeças)

Calculado a partir de classifier/data/*.pt (60 exames, apos auto-labeling
via classifier/auto_label.py, label_source=auto_cnn_limiar_duplo_v1),
contando apenas mini-epocas validas (REM, rswa_labels >= 0):

  n_valid_total = 615560 mini-epocas de 3s
  tonic:  positivos=33887  prevalencia=5.51%  pos_weight (neg/pos) = 17.17
  phasic: positivos=53549  prevalencia=8.70%  pos_weight (neg/pos) = 10.50
  any:    positivos=41418  prevalencia=6.73%  pos_weight (neg/pos) = 13.86

## Comando de treino recomendado (executar manualmente)

    cd /Users/fernandososter/Documents/sleep_staging_rswa_project
    python3 scripts/train_rswa.py \
      --data-dir classifier/data \
      --n-splits 5 \
      --test-fraction 0.2 \
      --epochs 30 \
      --patience 8 \
      --lr 1e-4 \
      --tonic-pos-weight 17.17 \
      --phasic-pos-weight 10.50 \
      --any-pos-weight 13.86 \
      --monitor rswa_f1_macro \
      --seed 42 \
      --device auto \
      --run-dir runs/rswa \
      --experiment-name rswa_3heads_auto_labels_v1 \
      --notes "treino com rotulos auto-gerados via CNN+limiar-duplo (auto_label.py); avaliacao provisoria até existir amostra_revisao revisada por humano"

## Estimativa de custo computacional (CPU, sem GPU disponivel localmente)

  - 1 passo de treino (forward+backward) em 1 exame de ~5.5h (menor exame,
    T=6590 mini-epocas): ~66s no modelo completo (D_MODEL=256, fallback
    GRU bidirecional pois mamba_ssm nao esta instalado).
  - Exame medio (T=10259, ~8.5h): ~103s/passo.
  - Epoca completa (48 sujeitos de treino em fold de 5, sem validacao): ~82 min.
  - Custo total aproximado do desenho acima (5 folds x ate 30 epocas, com
    early stopping patience=8): pode chegar a 100+ horas de CPU corridas.
  - Alternativas de menor custo, se necessario: reduzir --n-splits (3),
    --epochs (8-10) e --patience (3-4); ou setar D_MODEL=128 via variavel
    de ambiente antes de chamar o script (reduz o tempo por passo em ~1.6x,
    pois o gargalo é o scan sequencial do GRU/Mamba sobre a sequencia longa,
    nao o numero de parametros).
  - Alternativa recomendada: rodar em host remoto com GPU (torch.cuda ou,
    se mamba_ssm puder ser instalado, o kernel oficial do Mamba é
    tipicamente muito mais rapido que o fallback GRU para sequencias longas).

## Observacao sobre "LOSO-like"

Nao é LOSO completo (o que exigiria 60 folds de 1 sujeito cada, custo
proibitivo em CPU). StratifiedGroupKFold com n_splits=5 agrupa por sujeito
(nenhum sujeito aparece em treino E validacao no mesmo fold) e estratifica
por rotulo RSWA a nivel de mini-epoca -- e a aproximacao praticavel usada
pelo script já existente (scripts/train_rswa.py). Se quiser algo mais
proximo de LOSO, aumente --n-splits (ex.: 10 -> grupos de ~6 sujeitos por
fold), ao custo de mais tempo total de treino.
