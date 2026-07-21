# Sleep Staging + RSWA

Projeto Python extraído do notebook `Editing_Staging_RSWA_BiMamba_v1.ipynb`.

## Arquitetura

- `SleepStagingNet`: EEG/EOG → CNN multikernel → SE → BiMamba → W/N1/N2/N3/REM.
- `RSWADetectionNet`: EMG → CNN multikernel → SE → BiMamba → saídas tônica e fásica.
- `SleepStagingRSWASystem`: executa os dois modelos sem compartilhar pesos.
- A resolução temporal dos outputs é uma miniépoca de 3 segundos.

## Abrir no VSCode

```bash
unzip sleep_staging_rswa_project.zip
cd sleep_staging_rswa_project
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e .
cp .env.example .env
python scripts/inspect_models.py
pytest
```

Para CUDA/Mamba oficial, instale a versão de `mamba-ssm` compatível com seu PyTorch/CUDA. Sem ela, o projeto usa a implementação PyTorch incluída.

## Estado da migração

O código foi separado em configuração, dados, modelos, métricas e treinamento. O notebook original e uma exportação linear (`notebook_export.py`) foram mantidos para comparação.

O notebook original treinava e calculava F1/Kappa apenas para staging. O módulo `metrics.py` já contém métricas RSWA, mas a rotina de treinamento RSWA deve ser validada com os rótulos reais antes de uma execução longa.

## Dados esperados

Cada arquivo `.pt` deve representar um exame/sujeito e conter, no mínimo:

```python
{
    "subject_id": "S001",
    "signals": Tensor[T, 5, 300],       # 3 EEG + 1 EOG + 1 EMG
    "sleep_stages": Tensor[T],
    "rswa_labels": Tensor[T],       # 0=normal, 1=fásico, 2=tônico
    "rswa_conf": Tensor[T],
}
```

Por padrão, os canais 0–3 são usados pelo staging (3 EEG + 1 EOG) e o canal 4 pelo RSWA (EMG). Alternativamente, o EMG pode ser salvo separadamente nas chaves `emg_signals`, `emg` ou `emg_center`, com shape `[T,N]` ou `[T,1,N]`.

## Execução

Inspecionar e testar dimensões:

```bash
python scripts/inspect_models.py
python scripts/smoke_test.py
```

Treinar staging com divisão automática por sujeito:

```bash
python scripts/train_staging.py \
  --data-dir /caminho/tensors \
  --epochs 50 \
  --batch-size 1 \
  --device cuda
```

Treinar staging com diretórios já separados:

```bash
python scripts/train_staging.py \
  --train-dir /caminho/train \
  --val-dir /caminho/val
```

Treinar RSWA apenas em miniépocas REM:

```bash
python scripts/train_rswa.py \
  --data-dir /caminho/tensors \
  --epochs 30 \
  --min-confidence 0.0 \
  --device cuda
```

Treinar os dois ramos no mesmo DataLoader, mas com losses e otimizadores independentes:

```bash
python scripts/train_joint.py --data-dir /caminho/tensors --device cuda
```

Avaliar staging:

```bash
python scripts/evaluate.py \
  --task staging \
  --data-dir /caminho/test \
  --checkpoint checkpoints/staging/best.pt
```

Avaliar RSWA:

```bash
python scripts/evaluate.py \
  --task rswa \
  --data-dir /caminho/test \
  --checkpoint checkpoints/rswa/best.pt
```

Avaliar o sistema completo:

```bash
python scripts/evaluate.py \
  --task joint \
  --data-dir /caminho/test \
  --staging-checkpoint checkpoints/staging/best.pt \
  --rswa-checkpoint checkpoints/rswa/best.pt
```

Exportar as predições sincronizadas para CSV:

```bash
python scripts/predict.py \
  --data-dir /caminho/test \
  --staging-checkpoint checkpoints/staging/best.pt \
  --rswa-checkpoint checkpoints/rswa/best.pt \
  --output outputs/predictions.csv
```

## Logger de experimentos e ablation

Os scripts de treino criam uma pasta independente por execução em `runs/<task>/`.
Use `--experiment-name`, `--notes` e `--tags` para identificar as variantes.

Exemplo:

```bash
python scripts/train_staging.py \
  --data-dir /dados/tensors \
  --experiment-name no_mamba \
  --tags ablation staging no-mamba \
  --notes "Ablation removendo o bloco Mamba"
```

Cada run contém `run.json`, `data_split.json`, descrição do modelo, `metrics.csv`,
`metrics.jsonl`, `history.json`, `training.log`, `best.json`, `summary.json` e checkpoints.

## Validação cruzada estratificada e figuras

Os scripts `train_staging.py` e `train_rswa.py` usam `StratifiedGroupKFold` por padrão com 5 folds. A estratificação considera os rótulos das mini-épocas e o agrupamento mantém todas as mini-épocas de um sujeito no mesmo fold.

Executar todos os folds:

```bash
python scripts/train_staging.py --data-dir /caminho/tensors --n-splits 5
python scripts/train_rswa.py --data-dir /caminho/tensors --n-splits 5
```

Executar somente um fold:

```bash
python scripts/train_staging.py --data-dir /caminho/tensors --n-splits 5 --fold 2
```

Cada fold gera `training_curves.png`. Staging gera matrizes de confusão absoluta e normalizada da melhor época. RSWA gera matrizes separadas para tonic e phasic.


# COMANDOS PARA ABLATION
python scripts/train_staging.py \
    --data-dir /caminho/dos/dados \
    --model cnn

python scripts/train_staging.py \
    --data-dir /caminho/dos/dados \
    --model cnn_lstm \
    --lstm-hidden-size 128 \
    --lstm-layers 1

python scripts/train_staging.py \
    --data-dir /caminho/dos/dados \
    --model cnn_bilstm \
    --lstm-hidden-size 128 \
    --lstm-layers 1

python scripts/train_staging.py \
    --data-dir /caminho/dos/dados \
    --model cnn_bimamba