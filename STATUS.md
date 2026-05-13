# CasaIQ v3 — Status do Projeto

> Documento de progresso. Lê-se de cima pra baixo: o que é o projeto,
> como está organizado, o que já foi feito, e o que ainda falta.
> Última atualização: 2026-05-06 · 8 commits no `main` · 211 testes em ~1s.

---

## 1. O que é o CasaIQ

Sistema **local-first** de inventário doméstico inteligente. Fluxo típico:

1. Usuário fotografa um armário/caixa via web.
2. A foto entra num pipeline de 4 agentes que **segmenta**, **analisa**,
   **enriquece metadados** e **gera ícone** de cada objeto.
3. Tudo é catalogado num banco SQLite local com busca por palavras-chave.
4. Um chat permite perguntas em linguagem natural ("onde guardei a chave de
   fenda?") que viram SQL e retornam respostas faladas.

Princípio central: **roteamento inteligente entre Ollama local e Claude API**.
Tudo roda offline por padrão; Claude é acionado seletivamente
(segunda opinião quando confiança < 0.65, text-to-SQL, geração de respostas
de chat). 5 modos de operação configuráveis: `offline`, `local_primeiro`,
`hibrido`, `claude`, `inteligente` (padrão).

---

## 2. Arquitetura

```
                     ┌──────────────────┐
                     │   web/ (SPA 4    │
                     │   abas: HTML/JS) │
                     └────────┬─────────┘
                              │ HTTP
                     ┌────────▼─────────┐
                     │  api/ (FastAPI)  │
                     │  7 routers /api  │
                     └────────┬─────────┘
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
┌───────▼────────┐  ┌─────────▼─────────┐  ┌────────▼─────────┐
│  pipeline/foto │  │  agents/assistente│  │  pipeline/video  │
│  4 agentes em  │  │  /chat + text→SQL │  │  ffmpeg+frames   │
│  série + db    │  │  + RO connection  │  │  +dedup objetos  │
└───────┬────────┘  └─────────┬─────────┘  └────────┬─────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              │
                     ┌────────▼─────────┐
                     │     core/        │
                     │  roteador · llm  │
                     │  database · sql_ │
                     │  safe · logging  │
                     └────────┬─────────┘
                              │
                  ┌───────────┴────────────┐
                  │                        │
            Ollama (local)           Claude API
        qwen2.5vl, llama3.2      (fallback condicional)
```

**Defesa em camadas no `/chat`** (introduzida no commit `2750c0c`):

```
LLM gera SQL ─▶ validar_select() ─▶ garantir_limit() ─▶ conectar_readonly() ─▶ fetchall
                whitelist por        cap em LIMIT 100   SQLite mode=ro
                palavra inteira                         (engine recusa escrita)
```

---

## 3. Status por módulo

Legenda: ✅ pronto · 🟡 com dívida conhecida · ⚪ sem testes diretos · ⏳ pendente

### `core/` — núcleo

| Arquivo | Responsabilidade | Testes | Status |
|---|---|---:|---|
| `config.py` | constantes globais, `.env` | — | ✅ |
| `database.py` | conexão SQLite thread-safe, schema, seed | indireto | ✅ datetime adapters explícitos (Py 3.12) |
| `roteador.py` | decide Ollama vs Claude por modo/tarefa/confiança | **94** | ✅ instrumentado com logs JSON |
| `llm.py` | dispatcher Ollama/Claude com fallback | **22** | ✅ logs de fallback estruturados |
| `runtime.py` | troca de modelos em runtime (vision/text) | indireto | ⚪ sem testes diretos |
| `sql_safe.py` | validação + RO + introspecção de schema | **44** | ✅ defesa em 3 camadas |
| `logging_config.py` | JSONFormatter + setup_logging idempotente | **6** | ✅ stdlib only |

### `agents/` — pipeline lógico

| Arquivo | Responsabilidade | Testes | Status |
|---|---|---:|---|
| `agent_1_segmentador.py` | lista objetos da foto + bbox opcional | indireto | ⚪ testado via pipeline |
| `agent_2_analisador.py` | análise visual + 2ª opinião automática | indireto | ⚪ testado via pipeline |
| `agent_3_enriquecedor.py` | normaliza + categoriza + palavras-chave | indireto | ⚪ testado via pipeline |
| `agent_4_icone.py` | 4 estratégias em cascata (recorte/web/Claude/PIL) | indireto | ⚪ testado via pipeline |
| `assistente.py` | chat: text→SQL + execução RO + resposta | **15** | ✅ schema dinâmico via PRAGMA |

### `pipeline/`

| Arquivo | Responsabilidade | Testes | Status |
|---|---|---:|---|
| `ingestao.py` | processar_foto: 4 agentes + INSERT + ícone | **7** | ✅ thread-safe, logs JSON |
| `video.py` | processar_video: ffmpeg → frames → dedup | **10** | ✅ synthetic_id seguro |

### `api/`

| Arquivo | Responsabilidade | Testes | Status |
|---|---|---:|---|
| `app.py` | FastAPI app + lifespan + StaticFiles | smoke | ✅ migrado para `lifespan` |
| `schemas.py` | modelos Pydantic | indireto | ✅ `ChatRequest` com `max_length=1000` |
| `routes/localizacoes.py` | CRUD localizações | smoke | ✅ delete protegido |
| `routes/fotos.py` | upload + status | smoke | ✅ |
| `routes/videos.py` | upload + status | ⏳ | ⚪ sem smoke test próprio |
| `routes/objetos.py` | CRUD objetos | smoke | ✅ |
| `routes/chat.py` | POST /chat + GET /chat/historico | smoke | ✅ delega ao assistente |
| `routes/estatisticas.py` | totais agregados | smoke | ✅ |
| `routes/modelos.py` | lista modelos Ollama | smoke | ✅ |

### `web/` — frontend

| Arquivo | Responsabilidade | Testes | Status |
|---|---|---:|---|
| `index.html`, `app.js`, `style.css` | SPA com 4 abas | — | ⚪ sem testes E2E |

### CI / Build

| Arquivo | Responsabilidade | Status |
|---|---|---|
| `.github/workflows/test.yml` | pytest em push/PR | ⏳ workflow pronto, falta `git push` para repo no GitHub |
| `requirements-dev.txt` | deps de teste | ✅ |
| `pytest.ini` | pythonpath, filtros | ✅ |
| `tests/conftest.py` | fixtures (db_temp, set_modo, set_api…) | ✅ |

---

## 4. Trabalho realizado (cronologia)

| # | Commit | Objetivo | Δ testes |
|---:|---|---|---:|
| 1 | `7a96954` | **initial commit**: CasaIQ v3 + cobertura inicial (roteador, llm, pipeline foto, smoke) | +111 |
| 2 | `ce2144e` | **fix deprecations Py 3.12**: lifespan FastAPI + adapters datetime SQLite | 0 |
| 3 | `772f7c2` | **CI**: GitHub Actions rodando pytest em push/PR | 0 |
| 4 | `db99ad1` | **cobertura pipeline vídeo**: ffmpeg/dedup/synthetic_id/resiliência | +10 |
| 5 | `2750c0c` | **hardening /chat**: defesa em camadas (validar+RO+limits) | +54 |
| 6 | `79405b7` | **logging estruturado JSON** nas decisões do roteador | +31 |
| 7 | `f0ee274` | **schema dinâmico** no PROMPT_SQL via PRAGMA | +5 |
| 8 | `11ea85a` | **migração prints → logger** em 9 arquivos (consistência total) | 0 |

**Total**: 211 testes; suite roda em **1.0 a 1.2 segundos**; passa com
`pytest -W error::DeprecationWarning`.

---

## 5. Cobertura de testes — distribuição

```
tests/test_roteador.py           69 testes  (matriz de decisão de roteamento)
tests/test_sql_safe.py           44 testes  (validação SQL + RO + schema)
tests/test_roteador_logging.py   25 testes  (eventos JSON de decisões)
tests/test_llm.py                22 testes  (fallback Ollama↔Claude)
tests/test_assistente.py         15 testes  (chat + text→SQL + segurança)
tests/test_routers_smoke.py      13 testes  (smoke dos 7 routers)
tests/test_pipeline_video.py     10 testes  (vídeo: dedup, resiliência, etc.)
tests/test_pipeline_ingestao.py   7 testes  (foto: happy path + edge cases)
tests/test_logging_config.py      6 testes  (JSONFormatter + setup)
                              ─────────────
                                211 testes em <1.2s
```

> **Limitação atual**: contagem de testes não é cobertura real. Não medimos
> linha/branch coverage hoje. Adicionar `pytest-cov` ao CI com
> `--cov-fail-under=80` é item de pendência (curto prazo, #2). Suspeita
> baseada em inspeção: `agent_4_icone.py` provavelmente tem cobertura real
> abaixo de 50% apesar dos integration tests.

**Cobertura por área** (qualitativa):
- Lógica de roteamento e LLM dispatch: **forte** (~140 testes)
- Pipelines (foto + vídeo): **boa** (17 testes integration-style)
- Segurança do `/chat`: **forte** (44 + 15 = 59 testes)
- Logging: **boa** (31 testes)
- Routers HTTP: **smoke only** (13 testes — 1 happy + 1 erro por router)
- Agente 4 (cascata de 4 estratégias): **fraca** (só via pipeline)

---

## 6. O que está pronto (entregáveis observáveis)

✅ **Pipeline foto end-to-end**: foto → 4 agentes → DB com ícones, thread-safe via BackgroundTask.
✅ **Pipeline vídeo end-to-end**: vídeo → ffmpeg → keyframes → dedup por nome → DB.
✅ **`/chat` blindado**: SQL gerado por LLM passa por whitelist textual + LIMIT forçado + execução em conexão read-only. Vetores conhecidos (DROP, ATTACH, PRAGMA, multi-statement, comentários, CTE com DELETE) testados como bloqueados.
✅ **Schema dinâmico**: o LLM vê o estado real do banco em cada chamada — migrações futuras de schema não quebram silenciosamente o text-to-SQL.
✅ **Logging estruturado**: cada decisão do roteador, cada chamada de LLM, cada evento de pipeline é uma linha JSON única em stdout. Pipe-friendly (`tail -f | jq`).
✅ **CI workflow**: `.github/workflows/test.yml` pronto; ativa automaticamente no primeiro push pro GitHub.
✅ **Zero deprecation warnings**: suite roda com `-W error::DeprecationWarning`.

---

## 7. Pendências (priorizadas)

### Bloqueante (resolver antes de qualquer item abaixo)

| # | Item | Esforço | Por quê é bloqueante |
|---:|---|---:|---|
| 0 | **Push pro GitHub + ativar CI** | 15 min | O workflow `.github/workflows/test.yml` é teórico até o primeiro run real. Riscos não-óbvios: wheel de `ddgs` em ubuntu-latest, mudanças recentes do SDK `anthropic`, cache do pip mal calibrado. Sem essa validação, a confiança na suíte é só do ambiente local. |

### Curto prazo (≤ 2 h cada, alto valor)

| # | Item | Esforço | Valor | Pendente porque |
|---:|---|---:|---|---|
| 1 | **Backup automático do `casaiq.db`** | 1h | **alto** | DB é o coração do produto local-first. Solução: cron diário + `sqlite3 casaiq.db ".backup '/backup/casaiq_$(date +%F).db'"` (lida com WAL nativamente). |
| 2 | **`pytest-cov` no CI com `--cov-fail-under=80`** | 30 min | alto | Conta de testes ≠ cobertura. Provavelmente revela buracos concretos em `agent_4_icone.py` e nas branches de fallback do `core/llm.py`. |
| 3 | **Testes unitários do `agent_4_icone`** | 2h | alto | 4 estratégias em cascata com fallbacks condicionais (recorte→web→Claude→PIL). Único agente onde integration tests são insuficientes — outros 3 agentes ficam em médio prazo. |
| 4 | **`set_authorizer` no `conectar_readonly`** | 1h | alto | Bloquear `SQLITE_ATTACH`/`DETACH` no nível do engine via callback nativo. Camada 4 que faltava no `sql_safe` — hoje só o validador textual rejeita ATTACH. |
| 5 | **Bound em `historico_chat`** (DELETE com retenção) | 1h | médio | Tabela cresce sem teto. Impacto principal: tamanho do arquivo `.db` e INSERT performance (B-tree). `GET /chat/historico` já tem `LIMIT 20`, então leitura via UI não sofre. `historico_chat` está na blacklist do `schema_para_prompt`, então PRAGMA não regride. |
| 6 | **Smoke test para `/api/videos/ingerir`** | 30 min | médio | Único router sem cobertura própria. |
| 7 | **README de uso** | 1h | alto | Onboarding em outras máquinas. |
| 8 | **Pre-commit hook rodando pytest** | 30 min | médio | Reforça CI localmente. |

### Médio prazo (≤ 1 dia)

| # | Item | Esforço | Valor | Observação |
|---:|---|---:|---|---|
| 9 | **Migrações versionadas de schema** | 3h | médio | Hoje `init_db` mistura `CREATE IF NOT EXISTS` com `ALTER TABLE` ad hoc (coluna `progresso`). Solução proporcional: tabela `migrations(version, applied_at)` + arquivos `core/migrations/000N_*.sql` + runner de ~30 linhas. **Não usar Alembic** — overkill pro tamanho do projeto. |
| 10 | **Cobertura unitária dos agentes 1, 2, 3** | 3h | médio | Diferente do `agent_4`: lineares (1 prompt → JSON → retorno) e os integration tests pegam regressões. Adiar até `pytest-cov` mostrar buracos concretos. |
| 11 | **Endpoint `/api/observabilidade`** | 2h | médio | Consome os logs JSON e expõe agregações (decisões/h por modo/motivo). Fecha o loop da instrumentação — hoje o consumo é via `tail -f \| jq` (válido mas manual). **Pausar instrumentação adicional até este existir.** |
| 12 | **Checklist manual de testes UI** (`TESTING.md`) | 30 min | médio | Cobertura E2E sem dep extra (Playwright). 10 passos: upload foto, ver status mudar, abrir objeto, deletar, navegar abas, etc. |
| 13 | **Limpeza de nós isolados do grafo** | exploratório | baixo | Graphify reporta 77+ nós isolados; podem indicar código morto. |

### Roadmap maior (não planejado)

- **Cache de inferência LLM**: chamadas idênticas (prompt+modelo) reutilizam resposta. Reduz custo em batches grandes.
- **Multi-usuário** (autenticação leve): hoje é single-user local.
- **Mobile/PWA**: hoje a SPA web é desktop-first.
- **Backup/sync**: nenhum mecanismo de backup automático do `casaiq.db`.

---

## 8. Como rodar o projeto (TL;DR)

```bash
# 1. Ambiente conda (preferido) ou pip:
mamba activate casaiq        # ou: pip install -r requirements-dev.txt

# 2. Banco + diretórios criados na 1ª execução; opcional .env com ANTHROPIC_API_KEY

# 3. Servidor:
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
# abre http://localhost:8000

# 4. Testes:
pytest                          # 211 testes em ~1s
pytest -W error::DeprecationWarning   # rigoroso, mesmo do CI

# 5. Logs em produção:
uvicorn api.app:app 2>&1 | jq 'select(.logger=="casaiq.pipeline.foto")'
```

---

## 9. Stack técnica resumida

| Camada | Tecnologia | Versão | Por quê |
|---|---|---|---|
| Linguagem | Python | 3.12 | conda-forge, ambiente reprodutível |
| API | FastAPI + uvicorn | ≥0.115 | async, type hints, OpenAPI grátis |
| Banco | SQLite (WAL) | stdlib | local-first, zero ops |
| LLM local | Ollama | qwen2.5vl:7b, llama3.2:3b | sem GPU, 8 GB RAM |
| LLM cloud | Anthropic SDK | ≥0.40 | Claude Sonnet 4 |
| Imagens | Pillow | ≥11.0 | recortes + ícones PIL |
| Busca web | ddgs | ≥9.0 | imagens grátis sem API key |
| Vídeo | ffmpeg (subprocess) | sistema | extração de keyframes |
| Testes | pytest, pytest-mock | 9.0, 3.15 | parametrize + mocks finos |
| Observabilidade | logging stdlib | — | sem deps; JSON line-format |

---

## 10. Riscos e dívidas conhecidas

| Risco | Severidade | Mitigação atual |
|---|---|---|
| Sem backup automático do `casaiq.db` | **alta** ↑ | Pendente item #1. Promovido de média — DB é o coração do produto local-first; "usuário cuidar" é insuficiente |
| `ATTACH DATABASE` no engine read-only | **parcial** | Pendente item #4 (`set_authorizer`). Hoje só o validador textual rejeita; um bypass de regex anula a defesa |
| Cobertura real (linha/branch) desconhecida | média | Pendente item #2 (`pytest-cov`). Hoje só temos contagem de testes — pode mascarar buracos em `agent_4` e branches de fallback |
| CI nunca executou de verdade | **alta** ↑ | Pendente item #0 (push pro GitHub). Workflow YAML é teórico até o primeiro run real |
| Agente 4 (cascata de 4 estratégias) sem testes diretos | **média** ↑ | Pendente item #3. Lógica condicional não é coberta integralmente pelo pipeline |
| `historico_chat` cresce sem bound | **média** ↑ | Pendente item #5. Impacto: tamanho do arquivo `.db` e INSERT performance |
| Frontend sem testes E2E | **média** ↑ | Pendente item #12 (`TESTING.md` com checklist manual). Promovido de baixa — frontend é a única interface do usuário |
| Agentes 1, 2, 3 sem testes unitários diretos | baixa | Cobertos via pipeline integration tests; lógica linear |
| Schema sem migrações versionadas | média | Pendente item #9. Mudanças de schema hoje são ad hoc no `init_db` |
| ROCm AMD instável (CLAUDE.md) | aceito | Inferência usa CPU puro, modelos quantizados |
