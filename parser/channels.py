"""
Selecao e filtragem de arquivos EDF + resolucao de canais (tolerante a ausencias).

Convertido do notebook Parser_Exames (celula 12).
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import PSGConfig


# ─────────────────────────────────────────────────────────────────────────────
# Filtragem de arquivos EDF
# ─────────────────────────────────────────────────────────────────────────────

def is_raw_edf(path: Path) -> bool:
    """
    Retorna True apenas para arquivos EDF de sinal bruto (exclui scoring/anotacao).
    """
    if path.suffix.lower() != ".edf":
        return False

    name_lower = path.stem.lower()
    excluded_keywords = ["scor", "staging", "hypno", "annot", "event",
                         "label", "stage", "sleep_stage"]
    return not any(kw in name_lower for kw in excluded_keywords)


def list_raw_edfs(edf_dir: Path) -> List[Path]:
    """Lista apenas os EDFs brutos de um diretorio."""
    edf_dir = Path(edf_dir)
    all_edfs = list(edf_dir.glob("*.edf"))
    raw_edfs = [p for p in all_edfs if is_raw_edf(p)]
    skipped = len(all_edfs) - len(raw_edfs)

    print(f"[preprocess] {len(all_edfs)} arquivos .edf encontrados em {edf_dir.name}/")
    if skipped > 0:
        skipped_names = [p.name for p in all_edfs if not is_raw_edf(p)]
        print(f"             {skipped} ignorados (scoring/anotacao): "
              f"{skipped_names[:5]}{'...' if skipped > 5 else ''}")
    print(f"             {len(raw_edfs)} EDFs brutos para processar")

    return sorted(raw_edfs)


# ─────────────────────────────────────────────────────────────────────────────
# Resolucao de canais (tolerante a canais faltando)
# ─────────────────────────────────────────────────────────────────────────────

def find_channel(raw_ch_names: List[str], candidates: List[str]) -> Optional[str]:
    """
    Retorna o primeiro candidato encontrado nos canais do EDF
    (comparacao case-insensitive). None se nenhum for encontrado.
    """
    lower_map = {ch.lower(): ch for ch in raw_ch_names}
    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]
    return None


def resolve_channels(
    raw_ch_names: List[str],
) -> Tuple[List[Optional[str]], List[bool]]:
    """
    Tenta encontrar cada canal de CHANNEL_DEFS nos canais do EDF.

    Retorna
    -------
    matched : list[str | None]  — nome real no EDF ou None se ausente
    mask    : list[bool]        — True se o canal foi encontrado
    """
    matched: List[Optional[str]] = []
    mask: List[bool] = []
    for defn in PSGConfig.CHANNEL_DEFS:
        ch = find_channel(raw_ch_names, defn["candidates"])
        matched.append(ch)
        mask.append(ch is not None)
    return matched, mask


def find_mat_file(subject_id: str, scores_dir: Path) -> Optional[Path]:
    """
    Procura o hipnograma hyp_<subject><ext> em scores_dir.
    Aceita: -annot.fif, _sleepscoring.edf, -Hypnogram.edf, .mat
    """
    scores_dir = Path(scores_dir)
    for ext in ["-annot.fif", "_sleepscoring.edf", "-Hypnogram.edf", ".mat"]:
        p = scores_dir / f"hyp_{subject_id}{ext}"
        if p.exists():
            return p
    return None
