# Handoff Fedora offline — autonomia da Oracle

> Quando o Fedora for desligado (~14h amanhã 2026-06-01), a Oracle continua
> trabalhando sozinha por dias/semanas se preciso. Este doc é o "manual de
> retomada" pra quando o Fedora voltar.

## Estado migrado em 2026-05-31 ~15:30

### O que aconteceu antes do handoff:
- **Kennedy** crawler migrado do Fedora pra Oracle. Estado (URLs + jsonl atual) copiado pra `~/br_vtex_fedora_migrate/`. Tmux window `vtex_kennedy` continua de onde parou.
- **Minas** mesma coisa. Tmux `vtex_minas`.
- **Ali Fedora** parado (24 itens, IP flagado, sem ganho). Estado preservado em `~/br_vtex_fedora_migrate/data/export/aliexpress_fedora.jsonl`.
- **Auto-pipeline** rodando em `tmux main:auto_pipeline` — detecta término de cada batch e dispara o próximo.
- **Watchdog cron** instalado: roda a cada 5 min, escreve `~/casaiq/orchestrator/healthcheck.json`.

## O que está rodando autonomamente

```
tmux main:
├── ali_loop                    Ali Oracle loop com retry-on-failure
├── screwfix_extract            Screwfix UK Etapa B rate-limited
├── vtex_superpro               Super Pro Atacado (~27k pendentes)
├── vtex_delupo                 Delupo (~10k pendentes)
├── vtex_ferimport              Ferimport (~6k pendentes)
├── vtex_kennedy                Kennedy migrado (~22k pendentes)
├── vtex_minas                  Minas migrado (~12k pendentes)
├── batch_matting               18.370 imagens BR legadas → matted + 3 escalas
└── auto_pipeline               Orquestrador master

Cron: */5 * * * * watchdog.sh
```

## Cadeia automática que o auto_pipeline dispara

```
batch_01_matting (rodando, ETA ~6h)
   └─ termina →
batch_02_dinov2 (lança automático)
   └─ encoda 18k canônicos com DINOv2-small (~6h ARM)
   └─ produz embeddings.npy + embeddings_index.jsonl
   └─ termina →
batch_03_faiss
   └─ constroi gazetteer.faiss (~10 min)
   └─ FIM da pipeline base do gazetteer
```

Tempo total estimado: ~13h após o batch_01 terminar.

## Quando o Fedora ligar de novo

### 1. Verificar estado geral

```bash
~/projetos/casaiq/scripts/painel.sh
```

Mostra: tmux windows ativas, coleta por frente, batches em curso, recursos, healthcheck.

### 2. Puxar dados acumulados pro Fedora

```bash
~/projetos/casaiq/scripts/sync_oracle.sh pull-data       # jsonl de coleta
~/projetos/casaiq/scripts/sync_oracle.sh pull-results    # batches output

# Imagens matted (cuidado: pode ser GB)
rsync -avz oracle-casaiq:~/casaiq/batches/01_matting/output/ ~/dataset_matted/

# Embeddings + index (~150 MB, vale baixar)
rsync -avz oracle-casaiq:~/casaiq/batches/02_dinov2/embeddings.npy ~/projetos/casaiq/
rsync -avz oracle-casaiq:~/casaiq/batches/02_dinov2/embeddings_index.jsonl ~/projetos/casaiq/
rsync -avz oracle-casaiq:~/casaiq/batches/03_faiss/ ~/projetos/casaiq/data/gazetteer/
```

### 3. Verificar saúde

```bash
# Auto-pipeline log
ssh oracle-casaiq "tail -30 ~/casaiq/orchestrator/auto_pipeline.log"

# Healthcheck atual
ssh oracle-casaiq "cat ~/casaiq/orchestrator/healthcheck.json"

# Algum batch deu erro?
ssh oracle-casaiq "for b in ~/casaiq/batches/0*; do echo \"=== \$(basename \$b) ===\"; tail -5 \$b/errors.log 2>/dev/null; done"
```

### 4. Se algum tmux window morreu

O watchdog lista isso em `healthcheck.json` (campo `dead`). Pra restaurar:

```bash
# Exemplo: relançar vtex_kennedy
ssh oracle-casaiq "tmux new-window -t main -n vtex_kennedy \
  'CASAIQ_BRVTEX_BASE=/home/ubuntu/br_vtex_fedora_migrate CASAIQ_BRVTEX_SITE=kennedy CASAIQ_BRVTEX_DOMAIN=https://www.ferramentaskennedy.com.br CASAIQ_BRVTEX_PER_REQ_SLEEP=1.5 /home/ubuntu/miniconda3/bin/conda run -n deep_learning python -u /home/ubuntu/br_vtex/br_vtex_crawler.py extract --workers 2 2>&1 | tee /home/ubuntu/br_vtex_fedora_migrate/kennedy_oracle.log'"
```

(É resumable — vai continuar de onde parou.)

## Comandos úteis pra monitorar de FORA enquanto Fedora dorme

Do celular, qualquer terminal SSH na Oracle:

```bash
ssh oracle-casaiq
cd ~/casaiq
./painel.sh
```

## Risco residual + mitigação

| Risco | Mitigação |
|---|---|
| Oracle perder energia / restart | Tmux + nohup não sobrevivem. Mitigação: já está em ARM cloud com uptime alto. Recovery manual quando voltar. |
| Algum site flagar IP da Oracle | Watchdog detecta tmux ativo mas crawler retorna 0. Painel mostra. Pode pausar e tentar mais tarde. |
| batch_02 dinov2 morrer no meio | É restart-safe via pickle acumulativo. Auto-pipeline detecta e relança. |
| Disco encher | 125 GB livre, batch 01+02 usam ~10 GB. Folgado. |

## Próxima sessão (Fedora online)

- POC v2 com fixes (encoder 896, bbox cluster, fundo matted)
- Testar gazetteer real: foto coletiva + lookup FAISS + visualização
- Decidir se replica em CLIP em paralelo
- Integrar com agent_1 do CasaIQ (substituir Claude na identificação)
