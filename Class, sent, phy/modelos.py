"""Carregamento, binarização e limpeza das avaliações."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from . import SEED

# Nomes de coluna conhecidos (B2W-Reviews01, Olist e nomes genéricos)
CANDIDATOS_TEXTO = ["review_text", "review_comment_message", "texto", "text", "comentario", "review"]
CANDIDATOS_NOTA = ["overall_rating", "review_score", "nota", "rating", "score", "stars"]
CANDIDATOS_ROTULO = ["sentimento", "label", "rotulo", "sentiment"]

_MAPA_ROTULOS = {
    "positivo": 1, "pos": 1, "positive": 1, "1": 1,
    "negativo": 0, "neg": 0, "negative": 0, "0": 0,
}


def limpar_texto(txt: str) -> str:
    """Limpeza leve: minúsculas, remove URLs e normaliza espaços.

    Mantém palavras de negação e pontuação; não remove stopwords, porque
    'não', 'nunca' etc. invertem o sentido da frase.
    """
    txt = txt.lower()
    txt = re.sub(r"http\S+", " ", txt)
    txt = re.sub(r"\s+", " ", txt)
    return txt.strip()


def _detectar_separador(caminho: Path, encoding: str) -> str:
    """Escolhe o separador mais frequente na linha de cabeçalho."""
    with open(caminho, encoding=encoding, errors="replace") as f:
        cabecalho = f.readline()
    candidatos = [",", ";", "\t", "|"]
    return max(candidatos, key=cabecalho.count)


def _ler_csv(caminho: Path) -> pd.DataFrame:
    for encoding in ("utf-8", "latin-1"):
        try:
            sep = _detectar_separador(caminho, encoding)
            return pd.read_csv(caminho, sep=sep, encoding=encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não foi possível ler {caminho} como utf-8 nem latin-1.")


def _primeira_coluna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    existentes = {c.lower(): c for c in df.columns}
    for nome in candidatos:
        if nome in existentes:
            return existentes[nome]
    return None


def carregar_dados(
    caminho: str | Path,
    col_texto: str | None = None,
    col_nota: str | None = None,
    col_rotulo: str | None = None,
    amostra: int | None = None,
) -> pd.DataFrame:
    """Lê o CSV e devolve um DataFrame com as colunas `texto` e `rotulo` (1 = positivo, 0 = negativo).

    Se as colunas não forem informadas, tenta detectá-las pelos nomes mais comuns.
    Com nota (1-5): 4-5 = positivo, 1-2 = negativo, 3 é descartada (neutra/ambígua).
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Dataset não encontrado em '{caminho}'. Baixe um dataset de avaliações em "
            "português (ex.: B2W-Reviews01 ou Olist) e salve-o nesse caminho."
        )

    df = _ler_csv(caminho)
    col_texto = col_texto or _primeira_coluna(df, CANDIDATOS_TEXTO)
    if col_nota is None and col_rotulo is None:
        col_nota = _primeira_coluna(df, CANDIDATOS_NOTA)
        if col_nota is None:
            col_rotulo = _primeira_coluna(df, CANDIDATOS_ROTULO)

    if col_texto is None or (col_nota is None and col_rotulo is None):
        raise ValueError(
            f"Não identifiquei as colunas. Colunas do arquivo: {list(df.columns)}. "
            "Use --col-texto e --col-nota (ou --col-rotulo)."
        )

    if col_nota is not None:
        nota = pd.to_numeric(df[col_nota], errors="coerce")
        base = pd.DataFrame({"texto": df[col_texto], "nota": nota}).dropna()
        base = base[base["nota"] != 3].copy()
        base["rotulo"] = (base["nota"] >= 4).astype(int)
    else:
        bruto = df[col_rotulo].astype(str).str.strip().str.lower()
        rotulo = bruto.map(_MAPA_ROTULOS)
        base = pd.DataFrame({"texto": df[col_texto], "rotulo": rotulo}).dropna().copy()
        base["rotulo"] = base["rotulo"].astype(int)

    base["texto"] = base["texto"].astype(str).map(limpar_texto)
    base = base[base["texto"] != ""]
    # Textos repetidos em folds diferentes inflariam as métricas (vazamento)
    base = base.drop_duplicates("texto")

    if amostra and len(base) > amostra:
        base = base.sample(amostra, random_state=SEED)

    return base[["texto", "rotulo"]].reset_index(drop=True)
