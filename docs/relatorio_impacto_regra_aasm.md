# Relatorio de impacto: regra AASM (2023a) vs. regra atual de limiar duplo para RSWA

**Data:** 2026-08-10
**Escopo:** comparacao quantitativa entre a regra de producao (`auto_rswa.py` /
`threshold_rule.py`, limiar duplo/histerese contra baseline local rolante) e uma
nova implementacao isolada dos 3 criterios textuais da AASM (2023a) —
`src/sleep_rswa/preprocessing/aasm_rule.py` — em 5 exames reais com CSV revisado
por humano (`classifier/labels/*_revisado.csv`).

## 1. O que foi implementado

`aasm_rule.py` e um modulo **isolado** (nao importa nada de `auto_rswa.py` /
`threshold_rule.py` / `classifier/`) que reimplementa os 3 criterios da AASM
exatamente como especificados pelo usuario:

- **Tonica**: por epoca de 30s de estagio R, soma das duracoes de segmentos de
  amplitude ≥2× o nivel de atonia REM, cada segmento com duracao **estritamente
  maior que 5s**, cobrindo ≥50% da epoca (≥15s de 30s). Decisao no nivel da
  **epoca de 30s**, rasterizada para as 10 mini-epocas de 3s que a compoem.
- **Fasica**: por epoca de 30s dividida em 10 mini-epocas de 3s, ≥5 das 10
  (50%) devem conter ao menos um burst de 0.1–5.0s de amplitude ≥2× o nivel de
  atonia REM. Decisao tambem no nivel da epoca de 30s.
- **Any**: qualquer atividade ≥2× o nivel de atonia REM, independente da
  duracao (inclui segmentos de 5–15s que nao fecham nem tonico nem fasico).
  Decisao **por mini-epoca de 3s** e forcada, por construcao, a ser **superset**
  de tonico e fasico: toda mini-epoca marcada `tonic=1` ou `phasic=1` na
  rasterizacao tambem recebe `any=1` na epoca de 30s inteira.

Referencia de amplitude ("nivel de atonia REM"): percentil 10 do envelope RMS
(janela 100ms) do EMG de mento dentro das mini-epocas REM — o mesmo
`rem_baseline_uv` ja calculado por `rem_baseline.compute_rem_baseline` e
persistido no `.pt`. Um fallback para percentil 10 do envelope RMS dentro do
NREM (`compute_nrem_baseline_uv`) foi implementado para o caso em que o REM
nao tem atonia detectavel (RSWA tonica cobrindo todo o REM) — ver Secao 4.

## 2. Testes deterministicos

14 testes em `tests/test_aasm_rule.py`, com sinais sinteticos e amplitudes/
duracoes exatas nos limites de cada criterio:

- **Tonica** (4 testes): 2 segmentos >5s somando 16s/30s ativa; 1 segmento
  exatamente 5s (limite exclusivo) nao conta; segmentos >5s somando <50% nao
  ativa; amplitude abaixo do limiar nunca ativa.
- **Fasica** (4 testes): exatamente 5/10 mini-epocas com burst ativa; 4/10 nao
  ativa; burst fora da janela 0.1–5.0s nao conta; burst <0.1s nao conta.
- **Any** (3 testes): segmento entre 5–15s ativa `any` sem ativar tonico nem
  fasico; amplitude abaixo do limiar nao ativa; `any` e superset de tonico
  apos rasterizacao da epoca completa.
- **Integracao** (3 testes): epocas fora de R nunca sao rotuladas; fallback
  NREM e usado quando o baseline REM e invalido (NaN/0 epocas); `T` que nao e
  multiplo de 30s (10 mini-epocas) e rejeitado com `ValueError`.

Resultado: **14/14 testes passam** (suite completa do projeto: 32/32).

## 3. Comparacao quantitativa em exames reais

5 exames com CSV revisado por humano e REM suficiente para os 3 criterios
(`rbd1`, `rbd4`, `rbd9`, `ins2`, `n1`). Em todos os 5, o baseline REM (nivel de
atonia) e valido — nao houve caso de fallback NREM nesta amostra.

### 3.1 Contagens e % de epocas de 30s de estagio R positivas

| exam | label | n_rem_mini_epochs | n_rem_macro_epochs | n_mini_pos_atual | n_mini_pos_aasm | n_mini_pos_human | pct_macro_pos_atual | pct_macro_pos_aasm | pct_macro_pos_human |
|---|---|---|---|---|---|---|---|---|---|
| rbd1 | tonic | 1270 | 127 | 890 | 180 | 0 | 61.4 | 14.2 | 0.0 |
| rbd1 | phasic | 1270 | 127 | 29 | 1110 | 997 | 3.9 | 87.4 | 72.4 |
| rbd1 | any | 1270 | 127 | 563 | 1270 | 0 | 26.0 | 100.0 | 0.0 |
| rbd4 | tonic | 2930 | 293 | 1948 | 30 | 0 | 52.2 | 1.0 | 0.0 |
| rbd4 | phasic | 2930 | 293 | 49 | 2920 | 2987 | 3.4 | 99.7 | 67.9 |
| rbd4 | any | 2930 | 293 | 1269 | 2930 | 0 | 26.6 | 100.0 | 0.0 |
| rbd9 | tonic | 2060 | 206 | 546 | 10 | 130 | 24.8 | 0.5 | 0.0 |
| rbd9 | phasic | 2060 | 206 | 43 | 2020 | 1094 | 1.5 | 98.1 | 44.2 |
| rbd9 | any | 2060 | 206 | 456 | 2056 | 0 | 16.0 | 100.0 | 0.0 |
| ins2 | tonic | 2330 | 233 | 3987 | 0 | 0 | 5.2 | 0.0 | 0.0 |
| ins2 | phasic | 2330 | 233 | 101 | 2330 | 5696 | 0.4 | 100.0 | 15.5 |
| ins2 | any | 2330 | 233 | 838 | 2330 | 0 | 9.0 | 100.0 | 0.0 |
| n1 | tonic | 2390 | 239 | 586 | 40 | 0 | 23.4 | 1.7 | 0.0 |
| n1 | phasic | 2390 | 239 | 44 | 2020 | 1442 | 3.8 | 84.5 | 57.7 |
| n1 | any | 2390 | 239 | 604 | 2368 | 0 | 31.4 | 100.0 | 0.0 |

### 3.2 Matriz de concordancia (regra atual vs. AASM), por mini-epoca REM de 3s

| exam | label | n_rem_mini | atual_pos_aasm_pos(TP) | atual_pos_aasm_neg(FP) | atual_neg_aasm_pos(FN) | atual_neg_aasm_neg(TN) | agreement_pct | cohen_kappa |
|---|---|---|---|---|---|---|---|---|
| rbd1 | tonic | 1270 | 14 | 492 | 166 | 598 | 48.2 | -0.213 |
| rbd1 | phasic | 1270 | 6 | 0 | 1104 | 160 | 13.1 | 0.001 |
| rbd1 | any | 1270 | 107 | 0 | 1163 | 0 | 8.4 | 0.0 |
| rbd4 | tonic | 2930 | 29 | 952 | 1 | 1948 | 67.5 | 0.038 |
| rbd4 | phasic | 2930 | 10 | 0 | 2910 | 10 | 0.7 | 0.0 |
| rbd4 | any | 2930 | 269 | 0 | 2661 | 0 | 9.2 | 0.0 |
| rbd9 | tonic | 2060 | 5 | 358 | 5 | 1692 | 82.4 | 0.018 |
| rbd9 | phasic | 2060 | 4 | 0 | 2016 | 40 | 2.1 | 0.0 |
| rbd9 | any | 2060 | 104 | 0 | 1952 | 4 | 5.2 | 0.0 |
| ins2 | tonic | 2330 | 0 | 52 | 0 | 2278 | 97.8 | 0.0 |
| ins2 | phasic | 2330 | 1 | 0 | 2329 | 0 | 0.0 | 0.0 |
| ins2 | any | 2330 | 58 | 0 | 2272 | 0 | 2.5 | 0.0 |
| n1 | tonic | 2390 | 7 | 287 | 33 | 2063 | 86.6 | 0.013 |
| n1 | phasic | 2390 | 15 | 2 | 2005 | 368 | 16.0 | 0.001 |
| n1 | any | 2390 | 232 | 1 | 2136 | 21 | 10.6 | 0.001 |

### 3.3 Leitura dos numeros

- **`tonic`**: a regra AASM produz *muito menos* positivos que a atual em
  todos os 5 exames (ex.: rbd1 14.2% vs. 61.4% das epocas R; ins2 0% vs. 5.2%).
  A concordancia (accuracy) varia de 48% a 98%, mas o kappa de Cohen e proximo
  de zero ou **negativo** (rbd1: -0.213) — as duas regras discordam mais do que
  concordariam por acaso nesse subconjunto, porque tem definicoes
  estruturalmente diferentes (segmento >5s cobrindo ≥50% da epoca de 30s vs.
  histerese local por segmento isolado).
- **`phasic`**: a regra AASM marca **quase todas** as mini-epocas REM como
  fasicas (87–100% das epocas R), muito acima tanto da regra atual (0.4–3.9%)
  quanto do humano (15.5–72.4%). Kappa ≈ 0 em todos os casos — nao ha relacao
  estatistica com a regra atual.
- **`any`**: por construcao (superset) e pela saturacao de amplitude descrita
  na Limitacao 1, a AASM marca **100% das epocas R como `any=1` nos 5
  exames** (`pct_macro_pos_aasm` = 100.0 em todos). No nivel de mini-epoca
  individual a cobertura ja e quase total (98.9-100%: `n_mini_pos_aasm` /
  `n_rem_mini_epochs` = 1270/1270 em rbd1, 2930/2930 em rbd4, 2056/2060 em
  rbd9, 2330/2330 em ins2, 2368/2390 em n1). O humano nunca usou o rotulo
  `any` nestes 5 exames (Secao 4).

## 4. Limitacoes

1. **Saturacao do criterio de amplitude.** O nivel de atonia REM
   (`rem_baseline_uv`) e definido como o **percentil 10** do envelope RMS
   dentro do REM. Verificamos que a razao mediana/percentil-10 do envelope RMS
   dentro do REM e tipicamente **2.3×–2.7×** nestes exames — ou seja, a
   *mediana* da atividade normal do EMG de mento durante REM ja excede
   sozinha o limiar "≥2× o nivel de atonia" definido pela AASM. Resultado
   direto: aplicar o criterio de amplitude ao pe da letra contra um baseline
   de percentil 10 classifica ~50–65% das amostras de envelope como acima do
   limiar so por variabilidade normal do ruido/EMG basal, inflando
   artificialmente `phasic` e (por superset) `any` para quase 100% das epocas
   REM. Isso NAO e um bug de implementacao — e uma propriedade estatistica
   inerente a um limiar fixo contra um percentil baixo de uma distribuicao
   assimetrica (a maior parte da massa do envelope RMS fica acima do
   percentil 10 por definicao). A regra atual evita esse problema usando
   histerese contra um baseline **local rolante** (adaptativo por trecho, nao
   um unico percentil fixo do exame inteiro).
2. **Fallback NREM nao exercitado nesta amostra.** Nenhum dos 5 exames
   selecionados teve REM sem atonia detectavel (`rem_baseline_n_epochs=0` ou
   baseline invalido), entao `atonia_source="nrem_fallback"` nunca foi
   acionado fora dos testes sinteticos. O fallback esta implementado e
   testado deterministicamente (`test_nrem_fallback_used_when_rem_baseline_invalid`),
   mas seu comportamento em exames reais com RSWA tonica extrema (REM
   inteiramente sem atonia) permanece nao validado empiricamente.
3. **CSVs revisados nao cobrem `tonic` nem `any`.** Nos 5 exames analisados,
   os CSVs `classifier/labels/*_revisado.csv` contem quase exclusivamente
   eventos `type=phasic` (rbd9 e a unica exceao, com apenas 5 eventos
   `tonic`; nenhum evento `any` em nenhum dos 5). Isso significa que a
   concordancia com o humano so pode ser avaliada para `phasic` nesta amostra
   — nao ha ground truth humano para validar `tonic` ou `any` sob nenhuma das
   duas regras.
4. **Amostra pequena.** 5 exames de >100 disponiveis; suficiente para
   demonstrar a magnitude e a direcao do efeito (saturacao de `phasic`/`any`),
   mas insuficiente para uma estimativa robusta de sensibilidade/
   especificidade da regra AASM contra o humano.
5. **Segmentacao por limiar simples, sem histerese.** `aasm_rule.py` usa um
   corte binario simples (`envelope >= limiar`) para identificar segmentos,
   conforme a definicao textual da AASM (que nao especifica histerese). Isso
   torna a deteccao de segmentos mais sensivel a ruido pontual no envelope do
   que a regra atual (que usa k_on/k_off assimetricos), amplificando ainda
   mais o efeito da Limitacao 1.

## 5. Recomendacao

**Nao substituir a regra de producao pela implementacao literal da AASM sem
recalibrar a referencia de amplitude.** A implementacao dos 3 criterios esta
correta em relacao ao texto da norma (validada pelos 14 testes deterministicos
e pela propriedade de superset), mas o resultado pratico e inutil para
treinamento/anotação nesta forma: `phasic` e `any` saturam em quase 100% das
epocas REM em todos os exames testados, o que elimina o poder discriminativo
do rotulo.

Duas rotas de recalibracao, em ordem de esforco:

1. **Recalibrar o percentil do baseline de atonia** (ex.: usar percentil 25–50
   em vez de 10, ou uma estatistica mais robusta como a moda do envelope RMS
   em trechos already confirmados como atonicos) — mantendo a estrutura dos 3
   criterios da AASM intacta, apenas ajustando o denominador da razao 2×. Esta
   e a rota de menor esforco e a que primeiro deveria ser tentada.
2. **Adicionar histerese** ao `_segments_above_threshold` (k_on/k_off como na
   regra atual) para reduzir fragmentacao de segmentos por ruido pontual —
   pode reduzir parcialmente a saturacao de `phasic`, mas nao resolve a causa
   raiz (Limitacao 1), que e o proprio nivel do baseline.

Em qualquer caso, **qualquer recalibracao deve ser revalidada contra os
mesmos 5 exames (ou um conjunto maior) antes de qualquer uso em treinamento**,
comparando novamente contra os CSVs revisados — idealmente apos o time de
revisao anotar tambem eventos `tonic` e `any` explicitamente, hoje ausentes
das planilhas usadas para produzir a ground truth humana desta comparacao.

## 6. Status da integracao no pipeline (atualizado apos a Secao 5)

`rswa_source="aasm"` ja existe como **terceira opcao explicita** em
`preprocess_exam` / `preprocess.py` / `__main__.py` (CLI: `--rswa-source aasm
--aasm-atonia-pct <pct>`), coexistindo com `"csv"` e `"auto"` sem alterar o
comportamento destes dois (`"auto"` continua sendo o default; testado byte-a-
byte contra o smoke test anterior — mesmas contagens tonic/phasic/any).

- `label_exam_with_aasm_rule` (em `aasm_rule.py`) e o adaptador que converte a
  saida de `apply_aasm_rule` para o schema completo do `.pt`
  (`tonic_labels`/`phasic_labels`/`any_labels`/`rswa_labels`/`rswa_conf`/
  `tonic_cov`/`phasic_cov`/`any_cov`/`label_source`), compativel com o que
  `auto_label_rswa_from_signals` e `rasterize_rswa_annotations` ja produzem.
- O parametro `aasm_atonia_pct` (default `None` = usa `rem_baseline_uv`,
  percentil 10, tal como gravado no `.pt`) permite recalibrar o percentil do
  baseline de atonia **sem mudar a integracao** — passar, por exemplo,
  `aasm_atonia_pct=50` recalcula o nivel de atonia internamente com percentil
  50 em vez de 10. Testado ponta-a-ponta no exame sintetico de smoke test:
  percentil 10 satura (100% `any`/`phasic`), percentil 50 reduz `any` para
  ~43% das mini-epocas REM, percentil 90 zera todos os 3 criterios — confirma
  que o parametro se comporta como esperado e que a recalibracao da Secao 5
  (opcao 1) e so uma questao de escolher o percentil certo, nao de mudar
  codigo.
- `rem_baseline` (percentil 10 do envelope RMS em REM) passou a ser calculado
  **antes** da geracao de rotulos em `preprocess.py` (era depois), porque
  `rswa_source="aasm"` depende dele como entrada; isso nao afeta `"csv"` nem
  `"auto"`, que nao consomem `rem_baseline` na geracao de rotulos.
- **`aasm_rule.py` continua isolado** — nao importa nada de `auto_rswa.py` /
  `threshold_rule.py` / `classifier/`, e `rswa_source="aasm"` **NAO** e usado
  por nenhum script de lote nem e o default de `--rswa-source` — precisa ser
  passado explicitamente.
- **A recomendacao da Secao 5 permanece em vigor**: mesmo com a integracao
  pronta, `rswa_source="aasm"` **nao deve ser usado para reprocessar/re-
  anotar os >100 exames** ate a recalibracao do percentil (Secao 5, opcao 1)
  ser feita e revalidada contra os CSVs revisados. Rodar
  `preprocess_exam(..., rswa_source="aasm")` hoje, sem passar
  `aasm_atonia_pct` recalibrado, reproduz a saturacao da Secao 3/4 (confirmado
  no smoke test: 100/100 mini-epocas REM viram `phasic`+`any` com o exame
  sintetico, percentil 10 default). O aviso `AASM_CALIBRATION_WARNING` e
  impresso automaticamente no log sempre que `rswa_source="aasm"` e usado,
  como lembrete em tempo de execucao.

## 7. Arquivos gerados nesta analise

- `src/sleep_rswa/preprocessing/aasm_rule.py` — implementacao isolada dos 3
  criterios, incluindo o adaptador `label_exam_with_aasm_rule` usado pela
  integracao da Secao 6 (nao wireada em `__init__.py`).
- `src/sleep_rswa/preprocessing/preprocess.py`,
  `src/sleep_rswa/preprocessing/__main__.py` — adicionam `rswa_source="aasm"`
  e `aasm_atonia_pct`/`--aasm-atonia-pct` (Secao 6).
- `tests/test_aasm_rule.py` — 14 testes deterministicos.
- `comparacao_aasm_vs_atual_resumo.csv` — contagens e % de epocas positivas
  por exame/criterio (Secao 3.1).
- `comparacao_aasm_vs_atual_concordancia.csv` — matriz de concordancia
  atual-vs-AASM por mini-epoca REM (Secao 3.2).
- `comparacao_emg_mascaras_rbd1.png` — EMG + envelope + limiar AASM e as 3
  mascaras (atual, AASM, humano-revisado) num excerto REM de 15 min do exame
  `rbd1`, ilustrando a saturacao de `phasic`/`any` descrita na Limitacao 1.
