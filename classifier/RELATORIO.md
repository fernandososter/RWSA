# Detector de movimento — relatório

Detector binário de **movimento** (evento motor a noite toda) treinado nos 4 exames
anotados, para pré-anotar os 95 exames restantes. Código isolado em `classifier/`,
sem qualquer dependência de `src/sleep_rswa` — o único ponto de contato é o formato
dos arquivos `.pt` (entrada) e do CSV de anotações (saída).

## O que foi feito

- **Alvo**: rótulo binário `movimento = tônico ∪ fásico`, em **toda a noite** (sem máscara
  de REM). Confirmado na caracterização: a maioria dos eventos ocorre fora do REM.
- **Entrada**: canal de EMG do mento (canal 4 do `.pt`), z-scored por exame. A CNN vê
  uma janela de 5 mini-épocas (±2 de contexto = 15 s) e prediz o rótulo da mini-época central (3 s).
- **Modelo**: CNN pequena (~26k parâmetros) — stem multi-kernel com downsampling +
  3 blocos convolucionais. Sem Mamba/LSTM (decisão do usuário: CNN basta p/ padrões de EMG).
- **Treino**: focal loss + subamostragem de negativos (5:1) para o desbalanceamento (~10:1),
  early stopping por PR-AUC de validação.

## Desempenho esperado (validação leave-one-subject-out)

Cada linha é o resultado num sujeito que o modelo **nunca viu no treino** — é a melhor
estimativa honesta do que esperar nos 95 exames novos. Ponto de operação: **limiar 0,20**,
escolhido para alta cobertura (triagem).

| Sujeito | PR-AUC | Recall de evento | Falsos alarmes/h | Precisão (mini-época) |
|---------|:------:|:----------------:|:----------------:|:---------------------:|
| rbd1    | 0,66   | 93,3 %           | 35,1             | 0,30 |
| rbd2    | 0,31   | 80,0 %           | 16,1             | 0,30 |
| rbd3    | 0,60   | 96,1 %           | 29,4             | 0,34 |
| rbd5    | 0,38   | 98,0 %           | 36,3             | 0,13 |
| **Pooled** | **0,46** | **91,2 %**   | **~28**          | **0,28** |

**Como ler isto.** Como pré-anotador, o modelo captura ~90 % dos eventos de movimento
reais. O custo é ~28 marcações de falso-alarme por hora que o revisor humano descarta.
Ou seja: em vez de marcar do zero, você revisa uma lista já povoada e corrige — é onde
está o ganho de tempo. A precisão por mini-época é baixa (~28 %), esperado com apenas
4 sujeitos e alto desbalanceamento; para triagem o que importa é **não perder eventos**,
e o recall está bom.

## Limitações (honestas)

- **n = 4 sujeitos.** A variação entre folds é grande (PR-AUC 0,31–0,66). rbd2 e rbd5 são
  mais difíceis — rbd5 tem prevalência de só 3,4 % (outlier). Espere desempenho variável
  entre os 95, especialmente em exames com padrão de EMG diferente dos 4 de treino.
- **Falsos alarmes não são ruído puro.** Muitos "falsos" são provavelmente movimentos
  reais sub-limiar da sua anotação, ou bordas de eventos. Vale inspecionar antes de assumir erro.
- **Só EMG do mento.** Movimentos sem assinatura no mento (ex.: membros isolados) podem passar.
- O limiar 0,20 privilegia recall. Se quiser menos falsos alarmes (revisão mais rápida,
  arriscando perder eventos), suba o limiar — ver `threshold_sweep.png`.

## Como usar nos 95 exames

Pré-requisito: cada exame precisa estar no mesmo formato `.pt` do preprocessamento
(signals `[T, 5, 300]`, canal 4 = EMG do mento; rótulos não são necessários).

```bash
cd sleep_staging_rswa_project

# um exame -> CSV de anotações
python classifier/predict_movements.py caminho/EXAME.pt -o EXAME_movimentos.csv

# lote (todos os .pt de uma pasta)
for f in data/*.pt; do
  python classifier/predict_movements.py "$f" -o "${f%.pt}_movimentos.csv"
done
```

Saída — CSV com uma linha por evento, mesmo formato dos seus `*_rswa.csv`:

```
subject_id,onset_s,duration_s,type,score
rbd1,21.0,18.0,movement,0.4844
...
```

`type` é sempre `movement`; `score` é a confiança média do evento (útil para o revisor
priorizar — comece pelos de score alto). Exemplo real gerado: `rbd1_movimentos_exemplo.csv`
(305 eventos).

Opções úteis:
- `--threshold 0.3` — menos falsos alarmes, menos recall (default: 0,20, embutido no checkpoint).
- `--min-epochs 2` — descarta eventos de 1 mini-época (3 s), reduzindo ruído curto.
- `--model OUTRO.pt` — usar outro checkpoint.

## Re-treino (recomendado)

O modelo melhora com mais dados. Depois de revisar/corrigir os primeiros CSVs pré-anotados,
adicione esses exames a `classifier/data/` e rode:

```bash
python classifier/train_loso.py      # re-valida (opcional, mas recomendado)
python classifier/evaluate_loso.py   # re-escolhe o limiar
python classifier/train_final.py     # re-treina o modelo final
```

Com ~10–15 sujeitos o PR-AUC e a estabilidade entre folds devem subir de forma clara.

## Arquivos

```
classifier/
  movement_clf/         # pacote isolado (dataio, dataset, model, engine, metrics)
  train_loso.py         # validação leave-one-subject-out
  evaluate_loso.py      # métricas + escolha de limiar + figuras
  train_final.py        # treino final nos 4 exames -> movement_cnn_final.pt
  predict_movements.py  # INFERÊNCIA: novo .pt -> CSV de movimentos
  data/                 # os 4 .pt de treino (rbd1,2,3,5)
  outputs/              # modelo, métricas, figuras, CSV de exemplo
```
