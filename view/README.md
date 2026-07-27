# Revisor de movimento (`view/`)

Aplicação local para inspecionar as detecções do detector de movimento sobre o
EMG e confirmar cada evento como **tônico** / **fásico**, ou **apagar** falsos
positivos. Exporta um CSV revisado no **tempo do EDF** — pronto para re-treinar
o modelo a partir do tempo 0 do EDF.

> Isolamento: `view/` importa de `classifier/` (o detector), mas **nunca** de
> `src/sleep_rswa`. O contrato de isolamento é `classifier` ↔ `src/sleep_rswa`.

## Como funciona

1. Lê o canal **EMG direto do `.pt`** (o mesmo sinal que o modelo viu — alinhamento exato).
2. Roda o modelo (`classifier/outputs/movement_cnn_final.pt`) e marca as mini-épocas
   com score ≥ limiar; funde adjacentes em eventos.
3. Mostra o traçado de EMG com as mini-épocas de movimento **destacadas em laranja**;
   você navega evento a evento e rotula.
4. Salva `view/revisado/<exame>_revisado.csv` no formato dos seus `*_rswa.csv`
   (`subject_id, onset_s, duration_s, type, score`), com `type` ∈ {tonic, phasic}.

### Sugestão inicial
Cada evento já vem pré-rotulado a partir dos `tonic_labels`/`phasic_labels` do `.pt`
(quando existem). Eventos que o modelo detectou mas o `.pt` não marcava aparecem
como `movement` — são os candidatos novos a revisar.

### Dois modos de visualização
- **Eventos** — pula de evento a evento (o fluxo padrão de revisão).
- **Traçado completo** — rola a noite inteira do EMG contínuo, com controles de
  largura de janela (30s/60s/2min/5min) e uma régua para saltar a qualquer ponto.
  Clique num evento existente para selecioná-lo; clique-e-arraste num trecho sem
  marcação para criar um evento manual ali.

### Adicionar evento manualmente
Em qualquer modo, **+ Evento** (ou tecla `A`) cria um evento de 1 mini-época no
centro da janela atual; no modo Traçado completo, clique-e-arraste cria um evento
do tamanho do arraste. Eventos manuais não têm `score` do modelo (`score` sai
vazio no CSV) e entram na revisão como qualquer outro — ajuste com Tônico/Fásico
ou apague.

### Editar janelas dentro de um evento
O detector funde mini-épocas adjacentes marcadas em um único evento — então um
evento de, digamos, 18 s pode ter mini-épocas que não são movimento de verdade
misturadas com outras que são. O botão **✂ Editar janelas** (tecla `E`) liga um
modo em que clicar (ou arrastar) sobre o traçado alterna se aquela mini-época
fica **dentro** ou **fora** do evento selecionado (destacada em vermelho quando
excluída). **Restaurar evento** (tecla `R`) desfaz todas as exclusões do evento
atual de uma vez.

Ao salvar, cada evento é dividido nos trechos **contíguos** que restaram após as
exclusões — cada trecho vira sua própria linha no CSV. Um evento sem nenhuma
janela excluída sai como sempre (1 linha); se você excluir tudo, o evento não
gera linha nenhuma. Isso não muda o backend nem o formato do CSV — só decide
quais janelas de um evento entram como `onset_s`/`duration_s`.

## Tempo do EDF (necessário para o re-treino)

Os `onset_s` internos são relativos ao `.pt` (mini-época × 3 s). Seus `*_rswa.csv`
originais estão no **relógio do EDF** (t=0 = início do registro). A conversão é:

```
onset_edf = annot_start + onset_pt
annot_start = início_do_hipnograma − meas_date_do_EDF   (com correção de meia-noite)
```

`annot_start` é o instante do EDF onde começa a primeira época estagiada — o mesmo
ponto onde o preprocessamento cortou o sinal.

### Onde cada parte vem

- **início do hipnograma** — lido **automaticamente** de `view/mat/hyp_<exame>.mat`
  (campo `start_time`) quando você abre o exame. Basta ter os `.mat` na pasta.
- **meas_date do EDF** (hora em que o registro começou, ex. `22:47:30`) — você informa
  **na própria app**, no botão **⚙ Config**. É por exame e fica salvo em
  `view/exam_config.json`. O offset é calculado na hora.

Não há passo externo nem arquivo de offsets a pré-gerar: abriu o exame → digitou o
`meas_date` → o CSV já sai no tempo do EDF. Enquanto o `meas_date` não for informado,
a app funciona normalmente mas **avisa** ("⚠ sem offset") e o CSV sai em tempo do `.pt`.

> Onde achar o `meas_date`: é o horário no cabeçalho do EDF. Se não souber de cabeça:
> `python -c "import mne;print(mne.io.read_raw_edf('rbd1.edf',preload=False).info['meas_date'])"`
> (lê só o cabeçalho). O utilitário `compute_offsets.py` continua disponível para
> cálculo em lote pela linha de comando, mas não é necessário para o fluxo da app.

## Rodar a app

```bash
python view/app.py           # http://localhost:8000
python view/app.py --port 8080 --data classifier/data --model classifier/outputs/movement_cnn_final.pt
```

Abra o endereço no navegador. Atalhos: `←`/`→` navegar · `T` tônico · `F` fásico ·
`D` apagar. "Salvar CSV revisado" grava em `view/revisado/`.

## Arquivos

- `app.py` — servidor (biblioteca padrão + torch/numpy do venv) e API.
- `index.html` — interface (canvas do EMG, hipnograma da noite, botões de revisão, ⚙ Config).
- `mat/` — hipnogramas `hyp_<exame>.mat` (lidos para o início do estagiamento).
- `exam_config.json` — gerado pela app; `{exame: {meas_date, hipno_start, annot_start}}`.
- `compute_offsets.py` — utilitário opcional de linha de comando (cálculo em lote).
- `revisado/` — CSVs revisados de saída.


# Meas_date

## usando MNE
cd /Users/fernandososter/Documents/sleep_staging_rswa_project
source .venv/bin/activate
python - <<'EOF'
import mne
base="/Volumes/HD_EXTERNO/Workspace/AI/USP/dataset/capslpdb-1.0.0"
for ex in ["rbd1","rbd2","rbd3","rbd5"]:
    r = mne.io.read_raw_edf(f"{base}/{ex}.edf", preload=False, verbose="ERROR")
    md = r.info["meas_date"]
    print(f"{ex}: {md:%H:%M:%S}  ({md:%d/%m/%Y})")
EOF

## Usando Bash
dd if="/Volumes/HD_EXTERNO/Workspace/AI/USP/dataset/capslpdb-1.0.0/rbd1.edf" bs=1 skip=176 count=8 2>/dev/null