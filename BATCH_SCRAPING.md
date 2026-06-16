# 🤖 Batch Scraping Autônomo

## Resumo

Sistema de scraping **autônomo e sem supervisão** na Oracle VM que:
- ✅ Executa múltiplos spiders em **sequência**
- ✅ **Continua no próximo** site se um falhar
- ✅ Registra **cada erro** em logs detalhados
- ✅ Gera **relatório JSON** final com estatísticas
- ✅ **Não requer intervenção** humana

## Como Usar

### 1️⃣ INICIAR O BATCH (Oracle VM)

Via SSH do Fedora:

```bash
ssh oracle-casaiq "cd ~/casaiq_scraper && ./start_batch_scraping.sh"
```

Resposta esperada:
```
✅ Batch iniciado com PID: 12345
📋 Monitor com: tail -f /home/ubuntu/casaiq_scraper/logs/batch_current.log
```

### 2️⃣ MONITORAR PROGRESSO (Do Fedora)

**Ver status em tempo real:**
```bash
~/projetos/casaiq/scripts/monitor_batch.sh
```

Mostra:
- ✅ Status (RODANDO/FINALIZADO)
- 📋 Últimas linhas do log
- 📊 Relatórios já gerados

**Ou ver log contínuo:**
```bash
ssh oracle-casaiq "tail -f ~/casaiq_scraper/logs/batch_current.log"
```

### 3️⃣ RECUPERAR RELATÓRIO (Quando Terminar)

**Puxar relatório final e ver resumo:**
```bash
~/projetos/casaiq/scripts/fetch_batch_report.sh
```

Cria pasta `~/casaiq_batch_reports/` com:
- `batch_YYYYMMDD_HHMMSS.json` (relatório completo)
- Mostra tabela de resumo

**Ver relatório específico:**
```bash
cat ~/casaiq_batch_reports/batch_20260530_193000.json | jq '.'
```

---

## 📊 O Que o Script Faz

### Spiders Executados (em ordem)

```
1. obi.de (🇩🇪)           → ~400k produtos DIY
2. toolup.com (🇮🇪)       → ~200k ferramentas
3. cdiscount.com (🇫🇷)    → ~150k mixed
4. rscomponents (🇬🇧)     → ~200k eletrônicos
5. baumarkt.de (🇩🇪)      → ~50k ferramentas
6. toolstation.com (🇬🇧)  → ~80k ferramentas
7. screwfix.de (🇩🇪)      → ~100k ferramentas
8. heilind.com (🇺🇸)      → ~100k eletrônicos
9. brico.it (🇮🇹)         → ~30k mixed
10. ferreteria.net (🇪🇸)  → ~40k ferramentas
11. leroy_merlin.com.br (🇧🇷) → ~50k local
```

### Tratamento de Erros

| Evento | Ação |
|--------|------|
| Spider OK | ✅ Registra itens, continua próximo |
| Exit code ≠0 | ⚠️ Registra como PARTIAL, continua |
| Timeout (1-2h) | ⏱️ Registra TIMEOUT, continua |
| Erro crítico | ❌ Registra erro, continua |

**Nenhum spider "quebra" o batch** — sempre continua.

---

## 📁 Arquivos de Log

### Durante Execução
```
~/casaiq_scraper/
├── logs/
│   ├── batch_current.log           ← Log em tempo real (atualiza)
│   ├── batch_20260530_193000.log   ← Log final dessa execução
│   ├── spider_obi_20260530_193000.log
│   ├── spider_toolup_20260530_193000.log
│   └── ...
└── batch_reports/
    ├── batch_20260530_193000.json  ← Relatório estruturado
    └── ...
```

### Formato do Relatório JSON

```json
{
  "batch_id": "20260530_193000",
  "total_spiders": 11,
  "successful": 8,
  "partial": 2,
  "errors": 1,
  "total_items": 1234567,
  "duration_seconds": 28800,
  "results": [
    {
      "name": "obi",
      "status": "SUCCESS",
      "items": 412345,
      "duration_seconds": 3600,
      "error": null,
      "log_file": "..."
    },
    ...
  ]
}
```

---

## ⏱️ Tempo Estimado

| Fase | Spiders | Itens | Tempo |
|------|---------|-------|-------|
| **Total** | 11 | ~1.4M | **~10-12h** |
| Rápidos | 4 | ~650k | 4-5h |
| Médios | 4 | ~510k | 3-4h |
| Lentos | 3 | ~250k | 2-3h |

(Rodando 24/7 Oracle VM)

---

## 🔍 Troubleshooting

### Processo morreu antes de terminar
```bash
# Ver status
ssh oracle-casaiq "ps aux | grep python3"

# Ver últimas linhas do log
ssh oracle-casaiq "tail -100 ~/casaiq_scraper/logs/batch_current.log"

# Verificar se há um relatório parcial
ls ~/casaiq_batch_reports/
```

### Relatório não gerado
```bash
# Verificar se batch_reports existe
ssh oracle-casaiq "ls ~/casaiq_scraper/batch_reports/"

# Se não tiver, pode ter crashado — ver logs
ssh oracle-casaiq "tail ~/casaiq_scraper/logs/batch_*.log | grep ERROR"
```

### Cancelar batch em andamento
```bash
# Obter PID
ssh oracle-casaiq "cat ~/casaiq_scraper/.batch_orchestrator.pid"

# Matar processo
ssh oracle-casaiq "kill 12345"  # substitua 12345 pelo PID

# Confirmar
ssh oracle-casaiq "ps aux | grep 12345"
```

---

## 📈 Próximos Passos Após Batch

1. **Puxar dados:**
   ```bash
   rsync -avz oracle-casaiq:~/casaiq_scraper/data/export/ ~/projetos/casaiq/data/oracle_raw/
   rsync -avz oracle-casaiq:~/casaiq_scraper/data/raw/ ~/projetos/casaiq/data/raw_images/
   ```

2. **Enriquecer com Claude Haiku:**
   ```bash
   python3 ~/projetos/casaiq/scripts/enrich_dataset.py
   # (script ainda não existe — criar quando necessário)
   ```

3. **Push para HuggingFace Hub:**
   ```bash
   huggingface-cli upload cbpsoares/casaiq-dataset data.jsonl
   ```

4. **Fine-tunar no Colab:**
   - Abrir Google Colab
   - Carregar dataset de HF Hub
   - Fine-tune Qwen2-VL 7B com QLoRA
   - Export GGUF
   - Deploy no Ollama local

---

## 🎯 Por Que Funciona

✅ **Autônomo:** Não precisa de você acompanhando  
✅ **Resiliente:** Um erro não quebra tudo  
✅ **Loggado:** Cada erro é registrado para debug  
✅ **Rastreável:** JSON final tem todos os detalhes  
✅ **24/7:** Oracle VM tem energia garantida  

---

## 📞 Dúvidas?

Tudo o que você precisa saber está nos logs:
- `batch_reports/*.json` — resumo executivo
- `logs/batch_*.log` — timeline completa
- `logs/spider_*.log` — erros específicos

