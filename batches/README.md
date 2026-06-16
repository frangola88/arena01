# Batches — fábrica de dados na Oracle

Pré-processamento pesado que roda na Oracle em background (CPU 24/7).
Fedora escreve script → push pra Oracle → Oracle roda → pull resultados.

## Estrutura

```
batches/
├── README.md
├── _common/
│   └── batch_base.py        # classe BatchProcessor (restart-safe, status.json)
├── 01_matting/              # remoção de fundo + multi-escala (rembg/U2-Net)
│   ├── run.py
│   ├── input_list.txt       # 1 path por linha (gerado por find)
│   ├── status.json          # progresso atualizado a cada 50 itens
│   ├── log.jsonl            # 1 linha por item processado
│   ├── errors.log           # falhas humanlegíveis
│   └── output/              # <sha>.png + <sha>_{224,448,896}.jpg
└── (próximos: 02_dinov2_encoding, 03_faiss_index, ...)
```

## Padrão BatchProcessor

Cada batch herda de `_common/batch_base.BatchProcessor`. Implementa só `process(key) -> (bool, dict)`.
A base cuida de: restart-safe, status.json, log.jsonl, errors.log, progresso, SIGINT graceful.

## Como rodar

Na Oracle (em tmux pra não morrer com SSH disconnect):
```bash
tmux new-window -t main -n batch_NN \
  '/home/ubuntu/miniconda3/bin/conda run -n deep_learning python -u ~/casaiq/batches/NN/run.py 2>&1 | tee ~/casaiq/batches/NN/run.log'
```

## Monitor

```bash
ssh oracle-casaiq "cat ~/casaiq/batches/01_matting/status.json"
ssh oracle-casaiq "ls ~/casaiq/batches/01_matting/output | wc -l"
```

## Pull dos resultados pro Fedora

```bash
~/projetos/casaiq/scripts/sync_oracle.sh pull-results
# OU específico:
rsync -avz oracle-casaiq:~/casaiq/batches/01_matting/output/ ~/dataset_matted/
```
