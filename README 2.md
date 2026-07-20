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
