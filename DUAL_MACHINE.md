# Setup duas frentes — Fedora + Oracle

Documenta o que cada máquina faz e como sincronizar.

## Paridade técnica (estabelecida 2026-05-31)

| Componente | Fedora | Oracle |
|---|---|---|
| Python | 3.13 (sistema) + envs conda | 3.12.13 em `deep_learning` |
| torch | 2.4 (env casaiq) / 2.9 (ai-dl-rl) | **2.10.0+cpu** ARM |
| transformers | 5.x | **5.3.0** |
| sentence-transformers | (env llm tem) | ✅ instalado |
| faiss-cpu | (env llm tem) | ✅ instalado |
| matplotlib | global | ✅ instalado |
| DINOv2 testado | ✅ 0.4s/imagem | ✅ **0.24s/imagem 224x224** (ARM surpreendentemente rápido) |
| Ollama | sistema + qwen2.5:7b | systemd + llama3.2:3b |
| Hardware | Ryzen 8c/16t x86, 30GB RAM | ARM 4-core, 23GB RAM, 24/7 |

## Layouts espelhados

```
Fedora ~/projetos/casaiq/        ←→  Oracle ~/casaiq/
├── pipeline/                     ├── pipeline/         (mesmo Makefile + scripts)
│   ├── Makefile                  ├── poc_dinov2/       (mesmo poc.py)
│   ├── stage1_consolidate/       └── data/             (jsonl/parquet)
│   ├── stage2_imgs/
│   ├── stage3_enrich/
│   └── stage4_pack/
├── poc_dinov2/
├── scripts/sync_oracle.sh        ← helper de rsync
└── data/

Coleta legada continua em:
Oracle: ~/casaiq_scraper/, ~/br_vtex/, ~/ali_local/
```

## Divisão de trabalho recomendada

### Onde rodar O QUE (regra simples)

| Tarefa | Onde | Por quê |
|---|---|---|
| Scraping novos sites | **Oracle** (preferência) ou Fedora | IP US, 24/7, sobra de banda |
| Stage 1 consolidate | Fedora | Dataset final mora aqui |
| Stage 2 imagens download | **Oracle** | Network throughput melhor, 24/7 |
| Stage 3 Ollama Pass 1 | **Split** Fedora + Oracle | Fedora rápido (9h), Oracle paciente (56h) |
| Stage 3 Ollama Pass 2 | Fedora apenas | Cache de brand exige local único |
| Stage 4 pack | Fedora | Empacotamento + push HF |
| DINOv2 POC / experiments | Fedora (rapido) ou Oracle (background) | Ambos podem |
| Build gazetteer (embeddings DINOv2 dos 100k canônicos) | **Oracle** | 24/7, sobra CPU, pré-cálculo único |
| Inferência runtime CasaIQ (foto do user) | Fedora | Latência baixa, próximo do usuário |
| Tradução EN→PT (Screwfix) | Oracle | Background, llama3.2:3b já lá |
| Audit / sanity check 1% das classificações | Oracle | Background continuo |

## Comandos-chave de sincronização

```bash
# do Fedora:
~/projetos/casaiq/scripts/sync_oracle.sh status        # ver discrepâncias
~/projetos/casaiq/scripts/sync_oracle.sh push-code     # código Fedora → Oracle
~/projetos/casaiq/scripts/sync_oracle.sh pull-data     # jsonl Oracle → Fedora
~/projetos/casaiq/scripts/sync_oracle.sh pull-imgs     # imagens Oracle → Fedora
~/projetos/casaiq/scripts/sync_oracle.sh pull-results  # output POC/pipeline Oracle → Fedora
```

## Como rodar o POC DINOv2 na Oracle (idêntico ao Fedora)

```bash
# 1. já tem código sincronizado (push-code)
# 2. precisa picks.json — pode usar o mesmo do Fedora ou re-pickar lá com dados Oracle
ssh oracle-casaiq

# Dentro da Oracle:
cd ~/casaiq/poc_dinov2

# Re-pickar com dados disponíveis lá:
/home/ubuntu/miniconda3/bin/conda run -n deep_learning python <<PY
import json, random
from pathlib import Path
random.seed(7)
cands = []
for j in Path('/home/ubuntu/br_vtex/data/export').glob('*.jsonl'):
    for ln in open(j):
        d = json.loads(ln)
        u = d.get('img_url')
        if u and isinstance(u, str) and u.startswith('http') and d.get('nome_bruto'):
            cands.append(d)
print(f'total: {len(cands)}')
picks = []
for kw in ['furadeira','martelo','chave','serra','alicate']:
    m = [c for c in cands if kw in c['nome_bruto'].lower()]
    if m: picks.append(random.choice(m))
out = [{'idx':i,'nome':p['nome_bruto'][:80],'brand':p.get('brand'),'fonte':p.get('fonte'),'img_url':p['img_url']} for i,p in enumerate(picks[:5])]
Path('picks.json').write_text(json.dumps(out, ensure_ascii=False, indent=2))
PY

# Rodar POC:
/home/ubuntu/miniconda3/bin/conda run -n deep_learning python poc.py
```

## Próximos passos da paridade (a fazer)

- [ ] Refazer POC v2 em paralelo (Fedora 1 versão / Oracle outra com hiperparâmetros distintos)
- [ ] Pré-computar embeddings DINOv2 do gazetteer inteiro **na Oracle** (background ~10h)
- [ ] Index FAISS dos embeddings — gerar lá e baixar pro Fedora
- [ ] Quando Stage 1 fechar: lançar Stage 2 (imgs) na Oracle e Stage 3-pass1 sharded Fedora+Oracle
