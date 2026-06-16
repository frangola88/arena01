"""
Script para zerar completamente o banco de dados e arquivos relacionados.

USO:
    python scripts/resetar_banco.py             # confirma antes de apagar
    python scripts/resetar_banco.py --force     # apaga sem confirmar

O que faz:
1. Remove o arquivo do banco SQLite
2. Limpa diretórios: recortes/, icones/, fotos_originais/
3. Recria estrutura vazia com migrações + seed
"""
import sys
import shutil
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.config import DB_PATH, FOTOS_ORIGINAIS_DIR, RECORTES_DIR
from core.database import init_db


def confirmar() -> bool:
    """Pede confirmação ao usuário."""
    print("\n" + "=" * 60)
    print("  ATENÇÃO: ESTA AÇÃO É IRREVERSÍVEL!")
    print("=" * 60)
    print(f"\nSerão apagados:")
    print(f"  • Banco de dados: {DB_PATH}")
    print(f"  • Fotos originais: {FOTOS_ORIGINAIS_DIR}")
    print(f"  • Recortes: {RECORTES_DIR}")
    
    # Tentar localizar diretório de ícones
    icones_dir = DB_PATH.parent / "storage" / "icones"
    if not icones_dir.exists():
        icones_dir = Path.cwd() / "storage" / "icones"
    print(f"  • Ícones: {icones_dir}")
    
    print()
    resposta = input("Digite 'SIM' para confirmar: ").strip()
    return resposta == "SIM"


def limpar_diretorio(caminho: Path, descricao: str) -> int:
    """Limpa todos os arquivos de um diretório (mantém o diretório)."""
    if not caminho.exists():
        print(f"  ⚠ {descricao}: diretório não existe ({caminho})")
        return 0
    
    count = 0
    for arq in caminho.iterdir():
        if arq.is_file():
            arq.unlink()
            count += 1
        elif arq.is_dir():
            shutil.rmtree(arq)
            count += 1
    
    print(f"  ✓ {descricao}: {count} item(s) removido(s)")
    return count


def main():
    force = "--force" in sys.argv
    
    if not force and not confirmar():
        print("\n✗ Operação cancelada.")
        return 1
    
    print("\n🗑️  Iniciando reset do banco de dados...\n")
    
    # 1. Remover banco SQLite
    if DB_PATH.exists():
        DB_PATH.unlink()
        print(f"  ✓ Banco removido: {DB_PATH}")
    else:
        print(f"  ⚠ Banco não existia: {DB_PATH}")
    
    # 1b. Remover arquivos auxiliares do SQLite (WAL/SHM)
    for sufixo in ["-wal", "-shm"]:
        aux = Path(str(DB_PATH) + sufixo)
        if aux.exists():
            aux.unlink()
            print(f"  ✓ Removido auxiliar: {aux.name}")
    
    # 2. Limpar diretórios de arquivos
    limpar_diretorio(FOTOS_ORIGINAIS_DIR, "Fotos originais")
    limpar_diretorio(RECORTES_DIR, "Recortes")
    
    # Tentar limpar ícones (caminho varia)
    icones_dir = DB_PATH.parent / "storage" / "icones"
    if not icones_dir.exists():
        icones_dir = Path.cwd() / "storage" / "icones"
    limpar_diretorio(icones_dir, "Ícones")
    
    # 3. Recriar estrutura
    print("\n📦 Recriando estrutura do banco...")
    try:
        init_db()
        print("  ✓ Banco recriado com migrações e seed")
    except Exception as e:
        print(f"  ✗ Erro ao recriar banco: {e}")
        return 2
    
    print("\n" + "=" * 60)
    print("  ✅ RESET COMPLETO COM SUCESSO!")
    print("=" * 60)
    print("\n  O sistema está pronto para começar do zero.")
    print("  Reinicie o servidor para aplicar as mudanças.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
