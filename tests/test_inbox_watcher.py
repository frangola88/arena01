"""
Testes unitários do Inbox Watcher (scripts/inbox_watcher.py).
"""
import pytest
from pathlib import Path
from unittest.mock import patch
from scripts import inbox_watcher


@pytest.fixture
def inbox_env(tmp_path, monkeypatch):
    """Configura pastas temporárias para o inbox_watcher."""
    inbox_dir = tmp_path / "inbox"
    proc_dir = inbox_dir / "processados"
    err_dir = inbox_dir / "erros"
    fotos_dir = tmp_path / "fotos_originais"
    videos_dir = tmp_path / "videos_originais"

    monkeypatch.setattr(inbox_watcher, "INBOX_DIR", inbox_dir)
    monkeypatch.setattr(inbox_watcher, "PROCESSADOS_DIR", proc_dir)
    monkeypatch.setattr(inbox_watcher, "ERROS_DIR", err_dir)
    monkeypatch.setattr(inbox_watcher, "FOTOS_ORIGINAIS_DIR", fotos_dir)
    monkeypatch.setattr(inbox_watcher, "VIDEOS_ORIGINAIS_DIR", videos_dir)

    inbox_watcher.garantir_diretorios()

    return {
        "inbox": inbox_dir,
        "proc": proc_dir,
        "err": err_dir,
        "fotos": fotos_dir,
        "videos": videos_dir,
    }


def test_garantir_diretorios(inbox_env):
    for d in inbox_env.values():
        assert d.exists() and d.is_dir()


def test_listar_arquivos_pendentes_vazio(inbox_env):
    pendentes = inbox_watcher.listar_arquivos_pendentes()
    assert pendentes == []


def test_listar_arquivos_pendentes_com_arquivos(inbox_env):
    inbox = inbox_env["inbox"]
    foto = inbox / "foto_teste.jpg"
    foto.write_text("dummy")

    video = inbox / "video_teste.mp4"
    video.write_text("dummy")

    ignorado = inbox / "ignorado.txt"
    ignorado.write_text("dummy")

    pendentes = inbox_watcher.listar_arquivos_pendentes()
    caminhos = [p[0].name for p in pendentes]

    assert "foto_teste.jpg" in caminhos
    assert "video_teste.mp4" in caminhos
    assert "ignorado.txt" not in caminhos


def test_listar_arquivos_pendentes_subpasta(inbox_env):
    inbox = inbox_env["inbox"]
    sub = inbox / "Cozinha"
    sub.mkdir()
    foto = sub / "prato.png"
    foto.write_text("dummy")

    pendentes = inbox_watcher.listar_arquivos_pendentes()
    assert len(pendentes) == 1
    assert pendentes[0][0].name == "prato.png"
    assert pendentes[0][1] == "Cozinha"


def test_obter_ou_criar_localizacao(db_temp):
    # 1. Sem parâmetro -> retorna ID existente ou cria padrão
    loc1 = inbox_watcher.obter_ou_criar_localizacao("")
    assert loc1 >= 1

    # 2. Com parâmetro novo -> cria e retorna novo ID
    loc2 = inbox_watcher.obter_ou_criar_localizacao("Garagem Nova")
    assert loc2 > loc1

    # 3. Com o mesmo parâmetro -> reutiliza o ID existente
    loc3 = inbox_watcher.obter_ou_criar_localizacao("Garagem Nova")
    assert loc3 == loc2


def test_processar_arquivo_inbox_foto(inbox_env, db_temp, monkeypatch):
    monkeypatch.setattr(inbox_watcher, "arquivo_estavel", lambda path, intervalo_s=1.0: True)

    foto = inbox_env["inbox"] / "teste.jpg"
    foto.write_text("img_data")

    with patch("scripts.inbox_watcher.processar_foto") as mock_proc:
        sucesso = inbox_watcher.processar_arquivo_inbox(foto, "Escritório")
        assert sucesso is True
        mock_proc.assert_called_once()

    # Verifica se moveu para processados
    processados = list(inbox_env["proc"].glob("*_teste.jpg"))
    assert len(processados) == 1
    assert not foto.exists()


def test_processar_arquivo_inbox_video(inbox_env, db_temp, monkeypatch):
    monkeypatch.setattr(inbox_watcher, "arquivo_estavel", lambda path, intervalo_s=1.0: True)

    video = inbox_env["inbox"] / "clipe.mp4"
    video.write_text("vid_data")

    with patch("scripts.inbox_watcher.processar_video") as mock_proc:
        sucesso = inbox_watcher.processar_arquivo_inbox(video, "")
        assert sucesso is True
        mock_proc.assert_called_once()

    processados = list(inbox_env["proc"].glob("*_clipe.mp4"))
    assert len(processados) == 1
    assert not video.exists()
