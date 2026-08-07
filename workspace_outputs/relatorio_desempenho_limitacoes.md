# RSWADetectionNet (3 cabeças) -- Relatório curto de desempenho e limitações

**Data:** 2026-08-07
**Status:** entrega de infraestrutura completa (scripts + rotulagem automática + backup).
**Treino em escala real ainda não executado** -- ver seção "O que falta" abaixo.

## O que foi entregue nesta sessão

1. **Reversão da unificação de cabeças + adição da cabeça `any`** -- o modelo
   `RSWADetectionNet` volta a ter 3 saídas independentes (tônico, fásico, any),
   cada uma com seu próprio limiar de decisão.
2. **Rotulagem automática** (`classifier/auto_label.py`, CNN + limiar-duplo) --
   rodou nos 60 exames em `classifier/data/`, escrevendo direto nos `.pt`
   (label_source=`auto_cnn_limiar_duplo_v1`). Prevalências resultantes:
   tônico 5.51%, fásico 8.70%, any 6.73% (615.560 mini-épocas REM válidas).
3. **Backup de segurança** de todos os 60 `.pt` originais em
   `classifier/data_backup_auto_label/` (60/60 confirmados), feito ANTES da
   sobrescrita.
4. **Amostragem para revisão humana** (`classifier/sample_for_review.py`) --
   já gerou uma amostra por sujeito em `classifier/labels/amostra_revisao/`
   (60 arquivos `*_amostra_revisao.csv`), pronta para você revisar manualmente
   e assim criar o ground truth real.
5. **`scripts/evaluate_per_subject.py`** -- avaliação por sujeito e por cabeça
   (precisão/recall/F1/kappa), com dois modos: provisório (contra os próprios
   rótulos do `.pt`) e real (`--ground-truth-dir` apontando para CSVs
   revisados por humano, restringindo a avaliação às mini-épocas efetivamente
   amostradas).
6. **`scripts/predict_rswa.py`** -- inferência: `.pt` novo -> CSV
   (`onset_s, duration_s, type, score`), com fusão de mini-épocas adjacentes
   em eventos discretos por cabeça e suporte a ensemble de múltiplos folds.
7. **Smoke test de ponta a ponta** (8 sujeitos, D_MODEL=64, 4 folds x 2 épocas,
   ~35 min de CPU) validando que TODO o pipeline roda sem erro: treino →
   checkpoint → avaliação por sujeito → inferência → CSV de eventos. Não é
   uma medida de desempenho real (poucos dados, poucas épocas, apenas para
   testar os scripts).

## Por que as métricas de desempenho aqui são apenas ilustrativas

O relatório de avaliação por sujeito gerado nesta sessão
(`eval_smoke_report_provisorio.csv` / `eval_smoke_summary_provisorio.md`) usa o
checkpoint do **smoke test**, não de um treino real, e compara contra os
**próprios rótulos automáticos** do `.pt` (não contra revisão humana). Isso
tem duas consequências que tornam os números não confiáveis como medida de
desempenho:

- **Recall=1.000 em quase todas as linhas de tônico/any** é um artefato: com
  apenas 2 épocas e poucos dados, o modelo aprendeu a prever positivo quase
  sempre para essas cabeças (viés de threshold, não capacidade real de
  discriminação) -- daí precisão baixa (4-18%) e F1 baixo (0.06-0.31).
- **Cabeça fásico com F1=0 em quase todos os sujeitos:** o modelo previu
  negativo para quase toda mini-época nessa cabeça (ver
  `phasic_prediction_distribution` no checkpoint: 100% negativo). Combinado
  com prevalência real baixa (~8.7%), isso é o comportamento esperado de um
  modelo ainda não treinado de verdade -- convergiu para a classe majoritária
  na cabeça mais difícil.
- Comparar contra os próprios rótulos automáticos infla artificialmente
  qualquer concordância (o modelo pode estar aprendendo a imitar os erros
  sistemáticos do rotulador automático, não a detectar o evento real).

**Conclusão:** os números do smoke test confirmam que os scripts funcionam
corretamente (arquitetura, thresholding, agregação, geração de CSV), mas não
dizem nada sobre a qualidade real do detector. A avaliação real só existirá
depois de (a) treinar com o comando completo abaixo e (b) revisar
manualmente uma amostra de `classifier/labels/amostra_revisao/`.

## Limitações específicas da cabeça tônica

- **Tônus é um sinal de baixa frequência e alta persistência** (elevação
  sustentada de EMG por segundos-minutos), enquanto fásico é curto e
  transiente. O detector atual usa a mesma arquitetura/contexto temporal para
  as 3 cabeças; não há garantia de que a janela de contexto (mini-épocas de
  3s + vizinhança) seja suficiente para capturar elevações tônicas longas sem
  agregação adicional pós-hoc (a fusão de eventos em `predict_rswa.py` ajuda,
  mas opera sobre a saída do modelo, não sobre o sinal bruto).
- **Rotulagem automática do tônico é a mais heurística das três** -- o
  `auto_label.py` usa CNN + limiar-duplo, mas o critério AASM de "elevação
  sustentada de EMG" é mais sensível a normalização de amplitude
  (baseline por sujeito, ruído de eletrodo) do que fásico/any. Isso significa
  que os 5.51% de prevalência tônica no rótulo automático podem conter mais
  falsos positivos/negativos sistemáticos do que as outras cabeças -- só a
  revisão humana da amostra vai revelar isso.
- **Prevalência mais baixa (5.51%) e pos_weight mais alto (17.17)** tornam o
  tônico a cabeça mais sensível a overfitting/instabilidade em poucos dados
  ou poucas épocas -- exatamente o padrão observado no smoke test (viés para
  positivo).
- **Recomendação:** ao revisar a amostra humana, dar atenção extra aos
  eventos tônicos marcados pelo pipeline automático como "confirmados" -- essa
  é a cabeça onde a confiança na rotulagem automática é mais baixa.

## O que falta (ação do usuário)

1. **Rodar o treino real** (comando já pronto em
   `classifier/rswa_training_config.md`): 5 folds, 30 épocas, pos_weight
   calibrado por cabeça, D_MODEL=256 (o default do modelo -- NÃO usar 64, que
   foi só para o smoke test rápido). Estimativa: pode passar de 100h de CPU
   sem GPU; considere host remoto com GPU ou reduzir --n-splits/--epochs.
2. **Revisar manualmente** uma amostra dos CSVs em
   `classifier/labels/amostra_revisao/` para gerar o ground truth real.
3. **Re-rodar `evaluate_per_subject.py` com `--ground-truth-dir`** apontando
   para os CSVs revisados, usando o checkpoint do treino real (passe
   `--d-model 256`, que é o valor usado no treino recomendado).
4. **Selecionar o limiar de operação por cabeça** (tônico/fásico/any podem
   precisar de limiares diferentes de 0.5 -- o script aceita
   `--tonic-threshold`, `--phasic-threshold`, `--any-threshold`
   independentes) usando a curva precisão-recall da avaliação real.

## Arquivos relevantes desta entrega

- `classifier/auto_label.py` -- rotulagem automática (CNN + limiar-duplo)
- `classifier/data_backup_auto_label/` -- backup dos 60 `.pt` originais
- `classifier/auto_label_report.csv` -- relatório da rotulagem automática
- `classifier/sample_for_review.py` -- amostragem para revisão humana
- `classifier/labels/amostra_revisao/*.csv` -- amostra gerada (60 sujeitos)
- `classifier/rswa_training_config.md` -- pos_weight, comando de treino real
- `scripts/train_rswa.py` -- treino (3 cabeças, LOSO-like via StratifiedGroupKFold)
- `scripts/evaluate_per_subject.py` -- avaliação por sujeito/cabeça
- `scripts/predict_rswa.py` -- inferência (.pt novo -> CSV de eventos)
- `runs/rswa_smoke/20260807-102502_rswa_smoke_test/` -- checkpoints e logs do
  smoke test (NÃO usar para decisões de desempenho real)
