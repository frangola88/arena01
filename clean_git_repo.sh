#!/usr/bin/env bash

set -e

echo "======================================"
echo " Git repository cleanup for Python "
echo "======================================"

# -------------------------------------
# 1. Ensure we are inside git repo
# -------------------------------------

if ! git rev-parse --is-inside-work-tree > /dev/null 2>&1; then
    echo "Erro: este diretório não é um repositório Git."
    exit 1
fi

echo "Repositorio Git detectado."

# -------------------------------------
# 2. Create or update .gitignore
# -------------------------------------

echo "Criando/atualizando .gitignore..."

cat << 'EOF' > .gitignore

# ------------------------------
# Python cache
# ------------------------------
__pycache__/
*.pyc
*.pyo
*.pyd

# ------------------------------
# Virtual environments
# ------------------------------
.venv/
venv/
env/

# ------------------------------
# Build artifacts
# ------------------------------
build/
dist/
*.egg-info/

# ------------------------------
# Plots / outputs
# ------------------------------
*.png
*.jpg
*.jpeg
*.pdf

# ------------------------------
# Logs
# ------------------------------
*.log

# ------------------------------
# OS files
# ------------------------------
.DS_Store
Thumbs.db

# ------------------------------
# VSCode / Codespaces
# ------------------------------
.vscode/

EOF

echo ".gitignore atualizado."

# -------------------------------------
# 3. Remove cached compiled files
# -------------------------------------

echo "Removendo arquivos de cache do Git..."

git rm -r --cached __pycache__ 2>/dev/null || true
git rm -r --cached mini_stan/__pycache__ 2>/dev/null || true

# -------------------------------------
# 4. Remove compiled python artifacts
# -------------------------------------

find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

echo "Caches Python removidos."

# -------------------------------------
# 5. Remove generated plots from git index
# -------------------------------------

echo "Removendo imagens geradas..."

git rm --cached *.png 2>/dev/null || true
git rm --cached *.jpg 2>/dev/null || true

# -------------------------------------
# 6. Stage changes
# -------------------------------------

echo "Adicionando mudanças..."

git add .gitignore
git add .

# -------------------------------------
# 7. Commit
# -------------------------------------

echo "Criando commit..."

git commit -m "Repository cleanup: add gitignore and remove caches" || true

# -------------------------------------
# 8. Push
# -------------------------------------

echo "Enviando para GitHub..."

git push

echo "======================================"
echo "Repositório limpo e sincronizado."
echo "======================================"