# Como funciona o método testado
O teste compara duas formas de decidir, momento a momento, se o sinal EMG está "ativo" (evento) ou "em repouso" — a partir da mesma medida de amplitude e da mesma referência de fundo. A diferença entre as duas variantes está exclusivamente na regra de decisão sobre esse sinal já processado.

# Pre-Processamento 
## Etapa 0 — pré-processamento (igual para os dois métodos)

Envelope RMS: o EMG bruto é elevado ao quadrado e suavizado por uma janela deslizante de 0,1s, depois tira-se a raiz — isso dá uma curva de amplitude que sobe durante contração muscular e cai no repouso, sem a oscilação bipolar do sinal bruto.
Baseline local: em vez de um único valor fixo de "repouso" para o exame inteiro, calcula-se o percentil 10 do envelope dentro de uma janela local de 120s, deslizando ao longo do tempo. Isso segue o mesmo nível de ruído de fundo do paciente, que varia com a postura, o estágio de sono e a impedância do eletrodo ao longo da noite.
A partir daqui, os dois métodos diferem.

# Processamento 
## Variante 1 — limiar simples

Existe um único corte: o instante é "ativo" se envelope(t) > k · baseline(t), com k = 2,5. Assim que o envelope cruza essa linha, o evento está ligado; assim que cruza de volta para baixo, está desligado.

Problema estrutural conhecido: quando o envelope de um evento real oscila em torno desse único limiar (ruído de alta frequência somado a uma amplitude que está bem na fronteira do corte), o sinal cruza a linha várias vezes em rápida sucessão. O resultado é um evento sustentado sendo fragmentado em várias dezenas de segmentos curtos e espúrios — o que a literatura de detecção de sinal chama de "chattering". Foi exatamente isso que a medição confirmou: 1.418 fragmentos de ruído com menos de 0,3s, contra apenas 7 no método duplo, no conjunto de 10 exames sintéticos.

## Variante 2 — limiar duplo (histerese / trigger de Schmitt)

Aqui existem dois cortes, e a regra de transição usa cada um em um sentido diferente:

k_on = 3,0 (mais alto): é preciso cruzar esse corte para iniciar um evento — exige evidência forte de que algo realmente começou.
k_off = 1,5 (mais baixo): uma vez que o evento já está ativo, ele só termina quando o envelope cai abaixo desse corte mais baixo.
Tempo de desligamento configurável: opcionalmente, o sinal pode ser obrigado a permanecer abaixo de k_off por um tempo mínimo antes de o evento ser encerrado. Quedas breves abaixo desse corte são reabsorvidas como parte do mesmo evento.
Essa assimetria é o ponto central: para começar, o sinal precisa "provar" que é um evento real (limiar alto); mas para continuar sendo contado como o mesmo evento, ele só precisa não ter caído até o nível de repouso (limiar baixo). Pequenas flutuações de ruído que ficariam entre k_off e k_on — que fariam o limiar simples oscilar dentro e fora — não conseguem nem religar um evento já terminado (porque não chegam a k_on) nem desligar um evento em curso (porque não caem até k_off). É a mesma lógica usada classicamente em detecção de onset de EMG (Hodges & Bui) e em detecção de bordas em imagens (Canny), justamente para este fim.

# Pos-processamento

## Etapa final — pós-processamento e classificação (igual para os dois métodos)

Depois de qualquer um dos dois métodos gerar sua máscara ativo/inativo:

Fusão de lacunas curtas: qualquer intervalo inativo de até 1,0s entre dois segmentos ativos é fundido em um só — isso já absorve boa parte da fragmentação residual, inclusive a do limiar simples, o que é o motivo de o F1 final ter ficado praticamente idêntico entre os dois métodos no teste.
Classificação por duração: cada segmento resultante é rotulado por quanto tempo durou — fásico entre 0,5s e 5s, tônico a partir de 16s, e uma faixa intermediária ("zona morta") que não é classificada, pois no ground truth sintético ela só pode representar um erro de fragmentação/fusão do detector, nunca um evento real ambíguo.
