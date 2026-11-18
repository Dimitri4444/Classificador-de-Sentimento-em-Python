"""Validação cruzada estratificada e comparação entre os modelos."""

from __future__ import annotations

import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from . import SEED

METRICAS = {
    "accuracy": "accuracy",
    "precision_macro": "precision_macro",
    "recall_macro": "recall_macro",
    "f1_macro": "f1_macro",
    # A classe negativa costuma ser a minoritária e a mais importante para o negócio
    "f1_negativo": make_scorer(f1_score, pos_label=0),
    "f1_positivo": make_scorer(f1_score, pos_label=1),
}


def criar_cv(k: int = 5) -> StratifiedKFold:
    # Com random_state fixo, os folds são idênticos em todas as chamadas
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)


def validar(pipelines: dict[str, Pipeline], X: pd.Series, y: pd.Series, cv: StratifiedKFold) -> pd.DataFrame:
    """Roda a validação cruzada e devolve uma linha por (modelo, fold)."""
    linhas = []
    for nome, pipe in pipelines.items():
        r = cross_validate(pipe, X, y, cv=cv, scoring=METRICAS, n_jobs=-1)
        for fold in range(cv.get_n_splits()):
            linha = {"modelo": nome, "fold": fold + 1, "tempo_treino_s": r["fit_time"][fold]}
            linha.update({m: r[f"test_{m}"][fold] for m in METRICAS})
            linhas.append(linha)
    return pd.DataFrame(linhas)


def resumir(por_fold: pd.DataFrame) -> pd.DataFrame:
    """Média e desvio padrão de cada métrica, por modelo."""
    colunas = [c for c in por_fold.columns if c not in ("modelo", "fold")]
    resumo = por_fold.groupby("modelo")[colunas].agg(["mean", "std"])
    resumo.columns = [f"{m}_{estat}" for m, estat in resumo.columns]
    return resumo.reset_index()


def comparar_pareado(por_fold: pd.DataFrame, a: str = "TF-IDF", b: str = "BoW") -> pd.DataFrame:
    """Diferença (a - b) por métrica, fold a fold.

    Como os modelos usam os mesmos folds, a comparação é pareada. É uma
    análise descritiva: os folds compartilham dados de treino, então não
    tratamos a diferença como teste de significância.
    """
    pa = por_fold[por_fold["modelo"] == a].set_index("fold")
    pb = por_fold[por_fold["modelo"] == b].set_index("fold")
    linhas = []
    for m in METRICAS:
        d = pa[m] - pb[m]
        linhas.append({
            "metrica": m,
            f"dif_media_{a}_menos_{b}": d.mean(),
            "dif_min": d.min(),
            "dif_max": d.max(),
            f"folds_onde_{a}_vence": int((d > 0).sum()),
            "total_folds": len(d),
        })
    return pd.DataFrame(linhas)


def escolher_melhor(resumo: pd.DataFrame, metrica: str = "f1_macro") -> str:
    return resumo.loc[resumo[f"{metrica}_mean"].idxmax(), "modelo"]
