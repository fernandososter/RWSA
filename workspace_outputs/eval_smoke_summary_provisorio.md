# Avaliação por sujeito -- RSWADetectionNet (tônico/fásico/any)

**Modo:** PROVISÓRIO -- avaliado contra os rótulos do próprio `.pt` (mesmo pipeline automático que gerou os dados de treino; NÃO é avaliação independente). Substitua por `--ground-truth-dir` apontando para a amostra revisada por humano assim que ela existir.

Sujeitos carregados: 8 | avaliados: 8 | pulados (sem ground truth/sem mini-épocas válidas): 0

Limiares: tônico=0.5 fásico=0.5 any=0.5


## Cabeça: tonic

**Agregado (todos os sujeitos, todas as mini-épocas):** F1=0.139 | Precisão=0.074 | Recall=1.000 | Kappa=0.000 | n_positive=792/10640

Sujeitos com < 5 positivos (métricas instáveis): 0/8


| subject_id | n_mini_epochs | n_positive | precision | recall | f1 | kappa | instável |
|---|---|---|---|---|---|---|---|
| plm8 | 1820 | 234 | 0.129 | 1.000 | 0.228 | 0.000 |  |
| rbd1 | 1270 | 124 | 0.098 | 1.000 | 0.178 | 0.000 |  |
| ins4 | 1540 | 118 | 0.077 | 1.000 | 0.142 | 0.000 |  |
| rbd6 | 1290 | 113 | 0.088 | 1.000 | 0.161 | 0.000 |  |
| narco5 | 2160 | 90 | 0.042 | 1.000 | 0.080 | 0.000 |  |
| sdb1 | 1090 | 45 | 0.041 | 1.000 | 0.079 | 0.000 |  |
| plm5 | 1220 | 39 | 0.032 | 1.000 | 0.062 | 0.000 |  |
| plm6 | 250 | 29 | 0.116 | 1.000 | 0.208 | 0.000 |  |

## Cabeça: phasic

**Agregado (todos os sujeitos, todas as mini-épocas):** F1=0.000 | Precisão=0.000 | Recall=0.000 | Kappa=0.000 | n_positive=643/10640

Sujeitos com < 5 positivos (métricas instáveis): 1/8


| subject_id | n_mini_epochs | n_positive | precision | recall | f1 | kappa | instável |
|---|---|---|---|---|---|---|---|
| rbd1 | 1270 | 257 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| rbd6 | 1290 | 162 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| plm5 | 1220 | 72 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| narco5 | 2160 | 58 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| plm6 | 250 | 41 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| sdb1 | 1090 | 37 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| ins4 | 1540 | 15 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| plm8 | 1820 | 1 | 0.000 | 0.000 | 0.000 | 0.000 | ⚠ |

## Cabeça: any

**Agregado (todos os sujeitos, todas as mini-épocas):** F1=0.180 | Precisão=0.099 | Recall=1.000 | Kappa=0.000 | n_positive=1050/10640

Sujeitos com < 5 positivos (métricas instáveis): 0/8


| subject_id | n_mini_epochs | n_positive | precision | recall | f1 | kappa | instável |
|---|---|---|---|---|---|---|---|
| rbd1 | 1270 | 229 | 0.180 | 1.000 | 0.306 | 0.000 |  |
| narco5 | 2160 | 220 | 0.102 | 1.000 | 0.185 | 0.000 |  |
| rbd6 | 1290 | 203 | 0.157 | 1.000 | 0.272 | 0.000 |  |
| ins4 | 1540 | 114 | 0.074 | 1.000 | 0.138 | 0.000 |  |
| plm5 | 1220 | 111 | 0.091 | 1.000 | 0.167 | 0.000 |  |
| plm8 | 1820 | 74 | 0.041 | 1.000 | 0.078 | 0.000 |  |
| sdb1 | 1090 | 64 | 0.059 | 1.000 | 0.111 | 0.000 |  |
| plm6 | 250 | 35 | 0.140 | 1.000 | 0.246 | 0.000 |  |
