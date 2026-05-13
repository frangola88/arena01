# CasaIQ v3 — Inventário Doméstico Inteligente

Sistema **local-first** de catalogação de itens domésticos. Fotografa → Pipeline de IA → Chat em linguagem natural.

**Destaques:**
- 📷 4 agentes especializados (segmentação, análise, enriquecimento, geração de ícone)
- 🧠 Roteamento inteligente entre Ollama local e Claude API (paga apenas quando necessário)
- 🔒 100% offline por padrão; Claude acionado seletivamente para tarefas complexas
- 💾 SQLite local; zero dependências de cloud (exceto API Claude opcional)
- ⚡ 252 testes; suite em ~1.6s; 98% cobertura em componentes críticos

---

## 🚀 Início rápido (5 minutos)

### Pré-requisitos

- **Python 3.12.13** (ou 3.11+)
- **Ollama** rodando em `http://localhost:11434`
- **Modelos Ollama instalados:**
  ```bash
  ollama pull qwen2.5vl:7b    # visão (7B quantizado)
  ollama pull llama3.2:3b     # texto (3B quantizado)
  ```
- *(Opcional)* `ANTHROPIC_API_KEY` para roteamento com Claude

### Instalação com conda/mamba (recomendado)

```bash
# 1. Ativar ambiente conda
mamba activate casaiq    # ou: conda activate casaiq
# Se não existir, criar: mamba env create -f environment.yml

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Setup inicial
cp .env.example .env
# Editar .env se tiver ANTHROPIC_API_KEY

# 4. Rodar servidor
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload

# 5. Abrir em browser
# http://localhost:8000
```

### Instalação com venv (alternativa)

```bash
python -m venv venv
source venv/bin/activate          # ou: venv\Scripts\activate no Windows
pip install -r requirements.txt
cp .env.example .env
uvicorn api.app:app --reload
```

---

## 🎛️ Modos de operação

Controle via variável `CASAIQ_MODO` no `.env` ou código.

| Modo | Comportamento | Caso de uso |
|---|---|---|
| **`offline`** | 100% local, Claude desabilitado | Máxima privacidade; sem acesso à internet |
| **`local_primeiro`** | Ollama primeiro; Claude apenas como fallback | Otimiza custo; usa Claude quando confiança < 0.65 |
| **`hibrido`** | Visão local + texto/SQL/chat via Claude | Trade-off: privacidade + qualidade |
| **`claude`** | Tudo via Claude API | Máxima qualidade; requer API key |
| **`inteligente`** | **(padrão)** Roteador adapta por tarefa + confiança | Automático; recomendado |

Configurar em `.env`:
```bash
CASAIQ_MODO=inteligente
LIMIAR_CONFIANCA=0.65          # abaixo deste, usa Claude como 2ª opinião
```

---

## 📁 Estrutura do projeto

```
casaiq/
├── core/                   # Núcleo do sistema
│   ├── config.py          # Constantes, modo, limites
│   ├── database.py        # SQLite connection, schema, seed
│   ├── roteador.py        # Lógica de decisão Ollama vs Claude (94 testes)
│   ├── llm.py             # Dispatcher LLM com fallback (22 testes)
│   ├── sql_safe.py        # Validação + segurança SQL (44 testes)
│   └── logging_config.py  # JSONFormatter, logs estruturados
│
├── agents/                 # Pipeline de IA em 4 etapas
│   ├── agent_1_segmentador.py     # Detecta objetos + bbox
│   ├── agent_2_analisador.py      # Análise visual + 2ª opinião
│   ├── agent_3_enriquecedor.py    # Normaliza + categoriza
│   ├── agent_4_icone.py           # 4 estratégias de ícone (35 testes, 98% cobertura)
│   └── assistente.py              # Chat + text-to-SQL (15 testes)
│
├── pipeline/               # Processamento assíncrono
│   ├── ingestao.py        # foto → 4 agentes → DB (7 testes)
│   └── video.py           # vídeo → ffmpeg → frames → dedup (10 testes)
│
├── api/                    # FastAPI + routers
│   ├── app.py             # Lifespan, StaticFiles
│   ├── schemas.py         # Pydantic models
│   ├── routes/
│   │   ├── localizacoes.py    # CRUD localizações
│   │   ├── fotos.py           # Upload + processamento
│   │   ├── videos.py          # Upload vídeos (smoke tests)
│   │   ├── objetos.py         # CRUD objetos
│   │   ├── chat.py            # POST /chat + historico
│   │   ├── estatisticas.py    # Agregações
│   │   └── modelos.py         # Lista modelos disponíveis
│
├── web/                    # SPA (4 abas)
│   ├── index.html         # Estrutura + 4 abas
│   ├── app.js             # Lógica cliente
│   └── style.css          # Estilo
│
├── storage/               # Dados (gitignored)
│   ├── fotos_originais/   # Uploads
│   ├── recortes/          # Bounding boxes recortadas
│   └── icones/            # PNGs 256x256 gerados
│
├── .backups/              # Backups de DB (gitignored, auto-retenção 30d)
├── scripts/               # Utilitários
│   ├── backup_db.sh       # Backup automático
│   └── restore_db.sh      # Restauração de backup
├── tests/                 # 252 testes
├── STATUS.md              # Progresso do projeto (leia primeiro!)
├── BACKUP.md              # Guia de backup/restore
└── requirements.txt       # Dependências
```

---

## 🔧 Desenvolvimento

### Rodar testes

```bash
# Todos os testes
pytest                              # 252 testes em ~1.6s

# Teste específico
pytest tests/test_agent_4_icone.py -v
pytest tests/test_roteador.py::test_modo_offline -xvs

# Com cobertura
pytest --cov=core --cov-report=term-missing

# Modo rigoroso (como CI)
pytest -W error::DeprecationWarning
```

### Logs estruturados (JSON)

O sistema log em JSON por padrão. Filtrar:

```bash
# Logs da ingestão de foto
uvicorn api.app:app 2>&1 | jq 'select(.logger=="casaiq.pipeline.ingestao")'

# Decisões do roteador
uvicorn api.app:app 2>&1 | jq 'select(.logger=="casaiq.roteador")'

# Tudo
uvicorn api.app:app 2>&1 | jq '.'
```

### Database

```bash
# Inspecionar banco
sqlite3 casaiq.db ".schema"
sqlite3 casaiq.db "SELECT COUNT(*) FROM objetos;"

# Backup manual
sqlite3 casaiq.db ".backup '/tmp/casaiq_manual.db'"

# Restaurar
sqlite3 /tmp/casaiq_manual.db ".backup 'casaiq.db'"
```

### Variáveis de ambiente

```bash
# .env ou export
CASAIQ_MODO=inteligente                    # offline, local_primeiro, hibrido, claude, inteligente
LIMIAR_CONFIANCA=0.65                      # Threshold para fallback Claude
ANTHROPIC_API_KEY=sk-...                   # Se usar Claude
CASAIQ_PORT=8000                           # Porta do servidor
CASAIQ_HOST=0.0.0.0                        # Bind address
```

---

## 🎯 Fluxo de uso típico

1. **Ingerir foto**: 📸 Tire foto de um armário/caixa
   - Pipeline segmenta objetos → analisa cada um → enriquece metadados → gera ícone
   - Resultado: DB com objetos, ícone, categoria, palavras-chave

2. **Consultar**: 🔍 Procure por palavra-chave ou pergunta natural
   - "Onde guardei a chave de fenda?" → text-to-SQL → resultado em linguagem natural

3. **Organizar**: 📋 Veja inventário por localização/categoria
   - Editar, deletar ou mover objetos entre localizações

4. **Gerenciar**: ⚙️ Configure localizações e preferências
   - Criar/editar localizações (Sala, Garagem, etc.)

---

## 📊 Performance

- **Ingestão de foto**: ~2-5s (4 agentes em série)
- **Chat (text-to-SQL)**: ~1-2s (local) ou ~5s (com Claude)
- **Suite de testes**: ~1.6s (252 testes)
- **Armazenamento**: ~70KB banco inicial; cresce ~10KB por 100 objetos

---

## 🐛 Troubleshooting

### Ollama não está respondendo
```bash
# Verificar se está rodando
curl http://localhost:11434/api/tags

# Reiniciar
systemctl restart ollama    # Linux
brew services restart ollama # macOS

# Forçar modo offline
echo "CASAIQ_MODO=offline" >> .env
```

### Modelo não encontrado
```bash
# Listar modelos instalados
ollama list

# Baixar modelo
ollama pull qwen2.5vl:7b

# Definir em .env se mudar modelo
CASAIQ_MODELOS_VISAO=seu-modelo:tag
```

### Erro de permissão ao salvar ícone
```bash
# Verificar diretórios
ls -la storage/
chmod 755 storage/{fotos_originais,recortes,icones}
```

### Database corrompido
```bash
# Restaurar do backup mais recente
ls -lh .backups/
./scripts/restore_db.sh .backups/casaiq_YYYY-MM-DD_*.db
```

### Testes falhando
```bash
# Limpar cache do pytest
rm -rf .pytest_cache __pycache__ tests/__pycache__

# Verificar ambiente Python
python --version       # Deve ser 3.12+
pytest --version       # Deve ser 9.0+

# Reiniciar com pip
pip install -r requirements.txt --upgrade --force-reinstall
```

---

## 📚 Documentação adicional

- **[STATUS.md](STATUS.md)** — Progresso do projeto, pendências, cobertura de testes
- **[BACKUP.md](BACKUP.md)** — Setup de backup automático, restore, troubleshooting
- **[tests/](tests/)** — Exemplos de testes; rodar com `-xvs` para debug

---

## 🔐 Segurança

- ✅ SQL injection bloqueada: whitelist textual + LIMIT forçado + RO connection
- ✅ Sem senhas armazenadas: apenas ANTHROPIC_API_KEY em `.env` (gitignored)
- ✅ ATTACH/DETACH bloqueados no engine SQLite via `set_authorizer`
- ✅ Zero logging de dados sensíveis (apenas metadados em JSON)

---

## 📞 Acesso remoto (opcional)

Expor via Cloudflare Tunnel:

```bash
cloudflared tunnel --url http://localhost:8000
```

Acessar de qualquer lugar com URL segura (não é HTTPS puro, use apenas para testes).

---

## 📈 Próximos passos

Veja [STATUS.md](STATUS.md) para:
- Itens pendentes (curto/médio/longo prazo)
- Roadmap de features
- Bugs conhecidos

Sugestões: abra issue no repositório ou envie PR.

---

## 📜 Licença

Privado. CasaIQ é projeto pessoal de inventário doméstico.

---

**Última atualização:** 2026-05-13  
**Versão do projeto:** v3.0  
**Testes:** 252 passando | **Cobertura:** ~85% linha/branch  
**Status:** Em desenvolvimento ativo
