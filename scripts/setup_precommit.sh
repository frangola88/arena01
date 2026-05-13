#!/bin/bash
# Setup script — instala pre-commit hook no repositório local
#
# Uso:
#   ./scripts/setup_precommit.sh
#
# Isto copia o hook para .git/hooks/pre-commit e torna executável.
# O hook então roda pytest antes de cada commit.

set -euo pipefail

REPO_ROOT=$(git rev-parse --show-toplevel)
HOOK_SRC="$REPO_ROOT/scripts/pre-commit-hook.sh"
HOOK_DEST="$REPO_ROOT/.git/hooks/pre-commit"

if [[ ! -d "$REPO_ROOT/.git" ]]; then
    echo "ERRO: não estamos em um repositório git" >&2
    exit 1
fi

# Verificar se source existe
if [[ ! -f "$HOOK_SRC" ]]; then
    echo "ERRO: $HOOK_SRC não encontrado" >&2
    echo "Dica: este script deve ser rodado do repositório root"
    exit 1
fi

# Backup do hook existente se houver
if [[ -f "$HOOK_DEST" && ! -L "$HOOK_DEST" ]]; then
    echo "Backup do hook existente: $HOOK_DEST.bak"
    cp "$HOOK_DEST" "$HOOK_DEST.bak"
fi

# Copiar hook
cp "$HOOK_SRC" "$HOOK_DEST"
chmod +x "$HOOK_DEST"

echo "✓ Pre-commit hook instalado com sucesso"
echo "  Locação: $HOOK_DEST"
echo ""
echo "Próximos commits rodarão pytest automaticamente."
echo "Para bypass em emergências: git commit --no-verify"
