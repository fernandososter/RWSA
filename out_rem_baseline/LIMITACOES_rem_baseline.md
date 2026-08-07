# rem_baseline_uv — limitações conhecidas

## Campo

`rem_baseline_uv` / `rem_baseline_n_epochs`, gravados em todos os 60 `.pt`
canônicos (`classifier/data/`) e nas 41 cópias usadas pelos testes
determinísticos (`testes/data_real/`), calculado por
`src/sleep_rswa/preprocessing/rem_baseline.py`: percentil 10 do envelope RMS
do EMG bruto restrito às mini-épocas REM, convertido para µV assumindo que
`signals` está em Volts.

## Limitação: 3 exames com sinal EMG bruto em escala física diferente

`n13.pt`, `n14.pt`, `n15.pt` têm o canal EMG (e também EEG/EOG) armazenado
com desvio-padrão bruto na faixa ~20–90 "unidades", contra ~1e-6–1e-5 nos
demais 57 exames — uma diferença de escala de ordem 10^7. Isso indica que o
header do EDF de origem desses 3 exames foi interpretado com uma unidade
física diferente pelo parser (`mne.io.read_raw_edf`), não uma característica
real do sinal (não há saturação/clipping: apenas 1 amostra em cada extremo,
em 2,91M amostras — o sinal é contínuo e plausível na sua própria escala).

Consequência: `rem_baseline_uv` desses 3 exames (~2,1–3,1 milhões) **não
está em µV reais** e não é comparável aos demais 57 exames (que ficam entre
~0,21 e ~2,3 µV). Ver `rem_baseline_qc.png` — os 3 aparecem destacados no
painel (a) em escala log.

Causa raiz **não investigada** (decisão do usuário em 2026-08-06: manter
como está, documentar apenas, sem tocar no parsing de EDF nem no valor já
gravado). Ficam candidatos naturais para investigação futura caso o campo
seja promovido a uso ativo (ex. como limiar de detecção).

## Por que isso NÃO afeta o treino da rede neural

O `RSWADetectionNet` nunca consome o EMG bruto — consome a versão normalizada
por z-score **por exame** (`_zscore_per_channel` em `src/sleep_rswa/data.py`,
ou `zscore_emg` em `classifier/movement_clf/dataio.py`). Testado
empiricamente: `n13/n14/n15` e dois exames de escala normal (`rbd1`, `n1`)
todos chegam a `std == 1.0000` exato após a normalização, e não há
saturação/clipping em nenhum deles. A diferença de escala bruta é absorvida
pela normalização por-exame; os 3 exames **não precisam ser excluídos** do
treino/avaliação por este motivo.

## Escopo de uso atual

`rem_baseline_uv` está, por ora, **apenas persistido** — nenhum detector
(limiar/TKEO/CUSUM-GLR ou RSWADetectionNet) o consome ainda. Enquanto isso
for verdade, a limitação acima é inofensiva para o treino. Se e quando o
campo passar a ser usado (ex. como referência de baseline para os cabeçotes
tônico/fásico), os 3 exames devem ser tratados separadamente (excluídos do
uso desse campo específico, ou corrigidos na origem) antes de qualquer
decisão baseada nele.
