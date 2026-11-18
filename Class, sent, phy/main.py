"""Classificador de sentimento (positivo/negativo) para avaliações em português.

Uso:
    python main.py                       # lê dados/avaliacoes.csv, grava em resultados/
    python main.py --csv outro.csv --col-texto texto --col-nota nota

Gera todos os arquivos da pasta resultados/ (ver README.md).
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from sentimento import SEED
from sentimento.dados import carregar_dados
from sentimento.erros import (
    erros_por_segmento,
    exemplos_confiantes,
    matriz_confusao,
    previsoes_out_of_fold,
    relatorio_classificacao,
    termos_mais_pesados,
)
from sentimento.modelos import criar_pipelines
from sentimento.validacao import comparar_pareado, criar_cv, escolher_melhor, resumir, validar

DIR_RESULTADOS = Path("resultados")


def ler_argumentos() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="dados/avaliacoes.csv", help="caminho do dataset (padrão: %(default)s)")
    p.add_argument("--col-texto", help="coluna com o texto (detectada automaticamente se omitida)")
    p.add_argument("--col-nota", help="coluna com a nota de 1 a 5")
    p.add_argument("--col-rotulo", help="coluna já binária (positivo/negativo ou 1/0), se não houver nota")
    p.add_argument("--amostra", type=int, default=None, help="usa no máximo N avaliações (para testes rápidos)")
    p.add_argument("--folds", type=int, default=5, help="número de folds da validação cruzada")
    p.add_argument("--ngram-max", type=int, default=2, help="1 = unigramas; 2 = unigramas + bigramas")
    p.add_argument("--min-df", type=int, default=2, help="frequência mínima de um termo no vocabulário")
    return p.parse_args()


def main() -> None:
    args = ler_argumentos()
    DIR_RESULTADOS.mkdir(exist_ok=True)
    inicio = time.time()

    log: list[str] = []

    def registrar(texto: str = "") -> None:
        print(texto)
        log.append(texto)

    # 1. Dados ---------------------------------------------------------------
    df = carregar_dados(args.csv, args.col_texto, args.col_nota, args.col_rotulo, args.amostra)
    X, y = df["texto"], df["rotulo"]
    registrar("=== Dados ===")
    registrar(f"Arquivo: {args.csv}")
    registrar(f"Avaliações utilizadas: {len(df)} (positivas: {y.mean():.2%} | negativas: {1 - y.mean():.2%})")
    registrar(f"Python {platform.python_version()} | scikit-learn {sklearn.__version__} | "
              f"pandas {pd.__version__} | numpy {np.__version__} | seed {SEED}")

    # 2. Validação cruzada: BoW vs TF-IDF -----------------------------------
    pipelines = criar_pipelines(ngram_max=args.ngram_max, min_df=args.min_df)
    cv = criar_cv(args.folds)
    registrar(f"\n=== Validação cruzada estratificada ({args.folds} folds) ===")
    por_fold = validar(pipelines, X, y, cv)
    resumo = resumir(por_fold)
    pareado = comparar_pareado(por_fold)

    por_fold.round(5).to_csv(DIR_RESULTADOS / "cv_por_fold.csv", index=False)
    resumo.round(5).to_csv(DIR_RESULTADOS / "cv_resumo.csv", index=False)
    pareado.round(5).to_csv(DIR_RESULTADOS / "comparacao_pareada.csv", index=False)

    for _, linha in resumo.iterrows():
        registrar(f"\n{linha['modelo']}")
        for m in ("accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_negativo", "f1_positivo"):
            registrar(f"  {m:<16} {linha[f'{m}_mean']:.4f} ± {linha[f'{m}_std']:.4f}")
    registrar("\nComparação pareada (TF-IDF - BoW), fold a fold:")
    registrar(pareado.round(4).to_string(index=False))

    melhor = escolher_melhor(resumo, "f1_macro")
    registrar(f"\nMelhor modelo por F1-macro: {melhor}")

    # 3. Análise de erros do melhor modelo -----------------------------------
    registrar(f"\n=== Análise de erros ({melhor}, previsões out-of-fold) ===")
    previsoes = previsoes_out_of_fold(pipelines[melhor], X, y, cv)

    relatorio = relatorio_classificacao(previsoes)
    (DIR_RESULTADOS / "relatorio_classificacao.txt").write_text(
        f"Modelo: {melhor}\n\n{relatorio}", encoding="utf-8"
    )
    registrar(relatorio)

    matriz = matriz_confusao(previsoes)
    matriz.to_csv(DIR_RESULTADOS / "matriz_confusao.csv")
    registrar("Matriz de confusão (linhas = real, colunas = previsto):")
    registrar(matriz.to_string())

    segmentos = erros_por_segmento(previsoes)
    segmentos.round(4).to_csv(DIR_RESULTADOS / "erros_por_segmento.csv", index=False)
    registrar("\nTaxa de erro por segmento (dif_vs_taxa_geral > 0 = erra mais que a média):")
    registrar(segmentos[["grupo", "segmento", "n", "taxa_erro", "dif_vs_taxa_geral"]]
              .round(4).to_string(index=False))

    termos = termos_mais_pesados(pipelines[melhor], X, y)
    termos.round(5).to_csv(DIR_RESULTADOS / "termos_mais_pesados.csv", index=False)
    for classe in ("negativo", "positivo"):
        lista = ", ".join(termos.loc[termos["classe"] == classe, "termo"].head(15))
        registrar(f"\nTermos mais associados a '{classe}': {lista}")

    erros = previsoes[previsoes["erro"]].sort_values("confianca", ascending=False)
    erros[["tipo_erro", "confianca", "rotulo", "previsto", "n_palavras", "texto"]] \
        .round(4).to_csv(DIR_RESULTADOS / "erros.csv", index=False)
    exemplos_confiantes(previsoes).round(4).to_csv(DIR_RESULTADOS / "exemplos_erros_confiantes.csv", index=False)
    registrar(f"\nTotal de erros: {len(erros)} de {len(previsoes)} ({len(erros) / len(previsoes):.2%})")

    registrar(f"\nConcluído em {time.time() - inicio:.1f}s. Arquivos gravados em '{DIR_RESULTADOS}/'.")
    (DIR_RESULTADOS / "resumo.txt").write_text("\n".join(log) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
