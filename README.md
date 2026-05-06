# CasaIQ v3

Inventário doméstico inteligente com roteamento entre Ollama local e Claude API.

## Requisitos

- Python 3.11+
- Ollama rodando em `http://localhost:11434`
- Modelos: `qwen2.5vl:7b` (visão) e `llama3.2:3b` (texto)
- (Opcional) `ANTHROPIC_API_KEY` para roteamento inteligente

## Instalação rápida

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # ajuste ANTHROPIC_API_KEY se desejar
ollama pull qwen2.5vl:7b
ollama pull llama3.2:3b
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

Abrir `http://localhost:8000` no browser.

## Modos de operação (`CASAIQ_MODO`)

| Modo | Comportamento |
|---|---|
| `offline` | 100% local, Claude desabilitado |
| `local_primeiro` | Local; Claude apenas como fallback |
| `hibrido` | Visão local; texto/SQL/chat via Claude |
| `claude` | Tudo via Claude API |
| `inteligente` | Roteador decide por tarefa + confiança (recomendado) |

## Estrutura

```
core/      configurações, modelos, banco, roteador, executor de LLM
agents/    4 agentes (segmentar, analisar, enriquecer, ícone) + assistente
pipeline/  pipeline de ingestão (BackgroundTask thread-safe)
api/       FastAPI + 5 routers (localizacoes, fotos, objetos, chat, estatisticas)
web/       interface web (4 abas: Ingerir, Inventário, Assistente, Localizações)
storage/   fotos_originais, recortes, icones (criados na primeira execução)
```

## Acesso remoto opcional

```bash
cloudflared tunnel --url http://localhost:8000
```
