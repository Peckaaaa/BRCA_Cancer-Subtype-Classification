"""
PAM50 panel feature selection (domain-knowledge baseline).

Canonical version from `Code/MLP_benchmark.ipynb` (CELL 17). Matches the 50
PAM50 breast-cancer subtype genes (with a few legacy aliases) against feature
names formatted as "omics__gene". This is NOT data-driven: the same columns are
kept every fold. Defaults to matching mRNA only, with a fallback to all omics
when mRNA matches nothing.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

PAM50_GENES = [
    "ACTR3B", "ANLN", "BAG1", "BCL2", "BIRC5", "BLVRA", "CCNB1", "CCNE1", "CDC20", "CDC6",
    "CDH3", "CENPF", "CXXC5", "EGFR", "ERBB2", "ESR1", "EXO1", "FGFR4", "FOXA1", "FOXC1",
    "GPR160", "GRB7", "KRT14", "KRT17", "KRT5", "MAPT", "MDM2", "MELK", "MIA", "MKI67",
    "MLPH", "MMP11", "MYBL2", "MYC", "NAT1", "NOB1", "NUF2", "ORC6L", "PGR", "PHGDH",
    "PTTG1", "RRM2", "SFRP1", "SLC39A6", "TMEM45B", "TOP2A", "UBE2C", "UBE2T", "UHRF1", "WRN",
]

PAM50_ALIASES = {"ORC6L": "ORC6", "NUF2": "CDCA1"}

DEFAULT_PAM50_OMICS = ("mRNA",)


def _gene_token(fname: str) -> tuple[str, str]:
    """'mRNA__ESR1' -> ('mRNA', 'ESR1'); 'mRNA__ESR1|2099' -> ('mRNA', 'ESR1')."""
    omic, _, feat = fname.partition("__")
    gene = feat.split("|")[0].strip().upper()
    return omic, gene


def _aliases_of(gene_upper: str) -> set[str]:
    cands = {gene_upper}
    for k, v in PAM50_ALIASES.items():
        if gene_upper == k.upper():
            cands.add(v.upper())
        if gene_upper == v.upper():
            cands.add(k.upper())
    return cands


def coverage(feature_names: Sequence[str], omics: Optional[Iterable[str]]):
    """Return (selected_idx, found_genes, missing_genes, per_omic_count)."""
    present: dict[str, list[tuple[int, str]]] = {}
    for i, f in enumerate(feature_names):
        omic, gene = _gene_token(f)
        if omics is not None and omic not in omics:
            continue
        present.setdefault(gene, []).append((i, omic))

    selected_idx: list[int] = []
    found: list[str] = []
    missing: list[str] = []
    per_omic: dict[str, int] = {}
    for g in PAM50_GENES:
        hit = False
        for c in _aliases_of(g.upper()):
            for (i, omic) in present.get(c, []):
                selected_idx.append(i)
                per_omic[omic] = per_omic.get(omic, 0) + 1
                hit = True
        (found if hit else missing).append(g)
    return selected_idx, found, missing, per_omic


def select_indices(
    feature_names: Sequence[str], omics: Optional[Iterable[str]] = DEFAULT_PAM50_OMICS
) -> list[int]:
    idx, found, missing, per_omic = coverage(feature_names, omics)
    # Fallback: if mRNA matched nothing (e.g. Ensembl IDs), try every omics block.
    if len(idx) == 0 and omics is not None:
        idx, found, missing, per_omic = coverage(feature_names, None)
        if idx:
            print("  [PAM50] mRNA matched nothing -> fallback to ALL omics")
    if len(idx) == 0:
        raise RuntimeError(
            "PAM50: matched no genes. Feature names may not be gene symbols "
            "(e.g. Ensembl IDs) or use a different omics prefix."
        )
    omic_str = ", ".join(f"{k}:{v}" for k, v in sorted(per_omic.items()))
    print(f"  PAM50 selection     : matched {len(found)}/50 genes -> {len(idx)} features ({omic_str})")
    return idx
