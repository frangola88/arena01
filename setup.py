#!/usr/bin/env python3
"""Setup inicial do CasaIQ v3."""
import sys, subprocess, shutil
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()

def checar_python():
    v = sys.version_info
    if v < (3, 11):
        print(f"ERRO: Python 3.11+ necessário. Atual: {v.major}.{v.minor}")
        sys.exit(1)
    print(f"OK Python {v.major}.{v.minor}.{v.micro}")

def criar_venv():
    venv = BASE_DIR / "venv"
    if not venv.exists():
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    print("OK venv")
    return venv

def instalar_deps(venv):
    pip = venv/"bin"/"pip" if sys.platform != "win32" else venv/"Scripts"/"pip.exe"
    subprocess.run([str(pip), "install", "-r", str(BASE_DIR/"requirements.txt")], check=True)
    print("OK dependências")

def checar_ollama():
    if not shutil.which("ollama"):
        print("AVISO: Ollama não encontrado. Instale em https://ollama.com/download")
        print("       Ou configure ANTHROPIC_API_KEY no .env para usar Claude API.")
        return False
    try:
        r = subprocess.run(["ollama", "list"], capture_output=True, timeout=5)
        if r.returncode == 0:
            print("OK Ollama rodando")
            return True
    except Exception:
        pass
    print("AVISO: Ollama instalado mas não está rodando. Execute: ollama serve")
    return False

def baixar_modelos():
    print("\n=== Download dos modelos Ollama ===")
    print("AVISO — espaço necessário e tempo estimado:")
    print("  qwen2.5vl:7b  -> ~4.7 GB  (visão — 10-30 min dependendo da conexão)")
    print("  llama3.2:3b   -> ~2.0 GB  (texto — 5-15 min)")
    print("  RAM necessária: mínimo 8 GB (16 GB recomendado)")
    print("  GPU:  recomendada (sem GPU, cada foto leva 2-5 min)")
    print()
    if input("Baixar agora? (s/N) ").strip().lower() == "s":
        for m in ["qwen2.5vl:7b", "llama3.2:3b"]:
            print(f"\nBaixando {m}...")
            subprocess.run(["ollama", "pull", m])
    else:
        print("Baixe manualmente: ollama pull qwen2.5vl:7b && ollama pull llama3.2:3b")

def init_banco(venv):
    py = venv/"bin"/"python" if sys.platform != "win32" else venv/"Scripts"/"python.exe"
    subprocess.run(
        [str(py), "-c", "from core.database import init_db; init_db()"],
        cwd=str(BASE_DIR), check=True
    )
    print("OK banco inicializado")

def main():
    print("=== Setup CasaIQ v3 ===\n")
    checar_python()
    venv = criar_venv()
    instalar_deps(venv)
    if checar_ollama():
        baixar_modelos()
    init_banco(venv)
    ativ = "venv\\Scripts\\activate" if sys.platform == "win32" else "source venv/bin/activate"
    print(f"\n=== Pronto! ===")
    print(f"  {ativ}")
    print(f"  cp .env.example .env   # edite e adicione ANTHROPIC_API_KEY se desejar")
    print(f"  uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload")
    print(f"  Abrir: http://localhost:8000")

if __name__ == "__main__":
    main()
