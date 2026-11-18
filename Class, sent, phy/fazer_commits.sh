set -euo pipefail

[ -d .git ] || git init -b main

git add .gitignore requirements.txt dados/.gitkeep sentimento/__init__.py
git commit -m "chore: estrutura inicial do projeto, dependências e .gitignore"

git add sentimento/dados.py
git commit -m "feat(dados): carrega o CSV, binariza notas (4-5 pos, 1-2 neg) e limpa o texto"

git add sentimento/modelos.py
git commit -m "feat(modelos): pipelines BoW e TF-IDF com regressão logística balanceada"

git add sentimento/validacao.py
git commit -m "feat(validacao): validação cruzada estratificada e comparação pareada BoW x TF-IDF"

git add sentimento/erros.py
git commit -m "feat(erros): previsões out-of-fold, erros por segmento e termos mais pesados"

git add main.py
git commit -m "feat(main): orquestra o pipeline e grava todos os arquivos em resultados/"

git add README.md
git commit -m "docs: README com instruções, arquivos gerados e decisões de projeto"

git log --oneline
