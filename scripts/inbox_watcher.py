#!/usr/bin/env python3
"""
CasaIQ — Inbox Watcher Daemon (Ingestão Zero-Click)
Monitora storage/inbox/ e ingere fotos e vídeos automaticamente em segundo plano.

Uso:
  python scripts/inbox_watcher.py --once       # Processa arquivos pendentes e sai
  python scripts/inbox_watcher.py --daemon     # Fica rodando continuamente
  python scripts/inbox_watcher.py --status     # Mostra estatísticas da pasta inbox
"""

import sys
import os
import time
import shutil
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Garantir importação dos módulos do CasaIQ
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from core.config import STORAGE_DIR, FOTOS_ORIGINAIS_DIR, DB_PATH
from core.database import get_db
from pipeline.ingestao import processar_foto
from pipeline.video import processar_video

_log = logging.getLogger("casaiq.inbox_watcher")

INBOX_DIR = STORAGE_DIR / "inbox"
PROCESSADOS_DIR = INBOX_DIR / "processados"
ERROS_DIR = INBOX_DIR / "erros"
VIDEOS_ORIGINAIS_DIR = STORAGE_DIR / "videos_originais"

EXTENSOES_FOTO = {".jpg", ".jpeg", ".png", ".webp"}
EXTENSOES_VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
EXTENSOES_IGNORADAS = {".tmp", ".part", ".crdownload", ".download"}


def garantir_diretorios() -> None:
    """Cria a estrutura de pastas do inbox se não existir."""
    for d in [INBOX_DIR, PROCESSADOS_DIR, ERROS_DIR, FOTOS_ORIGINAIS_DIR, VIDEOS_ORIGINAIS_DIR]:
        d.mkdir(parents=True, exist_ok=True)


def obter_ou_criar_localizacao(nome_loc: str = "") -> int:
    """Retorna ID da localização pelo nome ou a primeira existente como padrão."""
    conn = get_db()
    try:
        if nome_loc and nome_loc.strip():
            nome_limpo = nome_loc.strip()
            row = conn.execute("SELECT id FROM localizacoes WHERE LOWER(nome) = LOWER(?)", (nome_limpo,)).fetchone()
            if row:
                return row["id"]
            # Criar nova localização baseada no nome da pasta
            cur = conn.execute(
                "INSERT INTO localizacoes (nome, tipo, comodo, descricao) VALUES (?, 'caixa', 'Geral', 'Criada automaticamente pelo Inbox Watcher')",
                (nome_limpo,)
            )
            conn.commit()
            _log.info("nova_localizacao_inbox", extra={"nome": nome_limpo, "id": cur.lastrowid})
            return cur.lastrowid

        # Fallback para a primeira localização existente
        row = conn.execute("SELECT id FROM localizacoes ORDER BY id ASC LIMIT 1").fetchone()
        if row:
            return row["id"]

        # Se nenhuma localização existe, cria a padrão
        cur = conn.execute(
            "INSERT INTO localizacoes (nome, tipo, comodo, descricao) VALUES ('Caixa de Entrada', 'caixa', 'Geral', 'Localização padrão de ingestão')"
        )
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def arquivo_estavel(caminho: Path, intervalo_s: float = 1.0) -> bool:
    """Verifica se o tamanho do arquivo parou de crescer (upload concluído)."""
    try:
        tam1 = caminho.stat().st_size
        if tam1 == 0:
            return False
        time.sleep(intervalo_s)
        tam2 = caminho.stat().st_size
        return tam1 == tam2
    except (FileNotFoundError, PermissionError):
        return False


def listar_arquivos_pendentes() -> list[tuple[Path, str]]:
    """
    Retorna lista de tuplas (Path, nome_subpasta_localizacao).
    Ignora pastas especiais como processados e erros.
    """
    pendentes = []
    if not INBOX_DIR.exists():
        return pendentes

    for item in INBOX_DIR.iterdir():
        if item.name.startswith(".") or item in (PROCESSADOS_DIR, ERROS_DIR):
            continue

        if item.is_file():
            ext = item.suffix.lower()
            if ext in (EXTENSOES_FOTO | EXTENSOES_VIDEO):
                pendentes.append((item, ""))
        elif item.is_dir():
            # Subpasta representa o nome da localização (ex: inbox/Cozinha/foto1.jpg)
            nome_localizacao = item.name
            for sub_item in item.iterdir():
                if sub_item.is_file() and not sub_item.name.startswith("."):
                    ext = sub_item.suffix.lower()
                    if ext in (EXTENSOES_FOTO | EXTENSOES_VIDEO):
                        pendentes.append((sub_item, nome_localizacao))

    return pendentes


def processar_arquivo_inbox(caminho: Path, subpasta_loc: str = "") -> bool:
    """Move o arquivo para a pasta definitiva, registra no DB e aciona o pipeline."""
    if not arquivo_estavel(caminho):
        _log.debug("arquivo_instavel_aguardando", extra={"arquivo": caminho.name})
        return False

    ext = caminho.suffix.lower()
    is_video = ext in EXTENSOES_VIDEO
    loc_id = obter_ou_criar_localizacao(subpasta_loc)

    # Gerar nome canônico único timestamped
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    novo_nome = f"{timestamp}{ext}"

    destino_dir = VIDEOS_ORIGINAIS_DIR if is_video else FOTOS_ORIGINAIS_DIR
    destino_path = destino_dir / novo_nome

    try:
        # Copiar para armazenamento canônico
        shutil.copy2(caminho, destino_path)

        conn = get_db()
        try:
            if is_video:
                cur = conn.execute(
                    "INSERT INTO videos_processados (caminho, localizacao_id, status) VALUES (?, ?, 'pendente')",
                    (str(destino_path), loc_id)
                )
                conn.commit()
                video_id = cur.lastrowid
                _log.info("iniciando_ingestao_video_inbox", extra={"video_id": video_id, "origem": caminho.name})
                processar_video(str(destino_path), loc_id, video_id)
            else:
                cur = conn.execute(
                    "INSERT INTO fotos_processadas (caminho, localizacao_id, status) VALUES (?, ?, 'pendente')",
                    (str(destino_path), loc_id)
                )
                conn.commit()
                foto_id = cur.lastrowid
                _log.info("iniciando_ingestao_foto_inbox", extra={"foto_id": foto_id, "origem": caminho.name})
                processar_foto(str(destino_path), loc_id, foto_id)
        finally:
            conn.close()

        # Mover original do inbox para processados
        arq_processado = PROCESSADOS_DIR / f"{timestamp}_{caminho.name}"
        shutil.move(caminho, arq_processado)
        _log.info("arquivo_inbox_concluido", extra={"arquivo": caminho.name, "destino": arq_processado.name})
        return True

    except Exception as e:
        _log.error("erro_processar_inbox", extra={"arquivo": caminho.name, "erro": str(e)}, exc_info=True)
        try:
            arq_erro = ERROS_DIR / f"{timestamp}_{caminho.name}"
            shutil.move(caminho, arq_erro)
        except Exception:
            pass
        return False


def ciclo_inbox(max_itens: int = 10) -> int:
    """Executa uma varredura do inbox processando até max_itens."""
    garantir_diretorios()
    pendentes = listar_arquivos_pendentes()
    if not pendentes:
        return 0

    sucesso = 0
    for caminho, subpasta in pendentes[:max_itens]:
        if processar_arquivo_inbox(caminho, subpasta):
            sucesso += 1
    return sucesso


def exibir_status() -> None:
    """Exibe estatísticas de arquivos no inbox."""
    garantir_diretorios()
    pendentes = len(listar_arquivos_pendentes())
    processados = len(list(PROCESSADOS_DIR.glob("*.*")))
    erros = len(list(ERROS_DIR.glob("*.*")))
    print("📥 ========================================================")
    print("📥 CasaIQ — Status do Inbox Watcher")
    print("📥 ========================================================")
    print(f"📁 Pasta Raiz:    {INBOX_DIR}")
    print(f"⏳ Pendentes:     {pendentes}")
    print(f"✅ Processados:   {processados}")
    print(f"⚠️  Com Erro:      {erros}")
    print("==========================================================")


def main() -> None:
    parser = argparse.ArgumentParser(description="CasaIQ Inbox Watcher")
    parser.add_argument("--once", action="store_true", help="Executa uma única varredura e sai")
    parser.add_argument("--daemon", action="store_true", help="Roda em loop contínuo")
    parser.add_argument("--intervalo", type=int, default=3, help="Segundos entre checagens no modo daemon (padrão: 3)")
    parser.add_argument("--status", action="store_true", help="Exibe contagem de arquivos e sai")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.status:
        exibir_status()
        return

    garantir_diretorios()
    print(f"👁️  CasaIQ Inbox Watcher ativo em: {INBOX_DIR}")

    if args.once:
        count = ciclo_inbox()
        print(f"✅ Varredura concluída. {count} arquivo(s) processado(s).")
        return

    # Modo daemon padrão
    print(f"🔄 Modo Daemon iniciado (checagem a cada {args.intervalo}s). Pressione Ctrl+C para encerrar.")
    try:
        while True:
            ciclo_inbox()
            time.sleep(args.intervalo)
    except KeyboardInterrupt:
        print("\n🛑 Inbox Watcher encerrado pelo usuário.")


if __name__ == "__main__":
    main()
