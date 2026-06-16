# Pipeline CasaIQ — Etapas 1-4. Custo R$ 0 (Ollama local).
# Roda do Fedora. Etapa 2 também pode rodar do Oracle (mais paralelismo de IP).

OLLAMA_MODEL ?= llama3.2:3b
HF_REPO ?= cbpsoares/casaiq-tools-v1

.PHONY: help all stage1 stage2 stage3 stage3-pass1 stage3-pass2 stage4 stage4-push stats clean

help:
	@echo "Pipeline CasaIQ — alvos disponíveis:"
	@echo "  make stage1            # consolida + dedupe (rsync Oracle → parquet)"
	@echo "  make stage3-pass1      # classifica util_casaiq + categoria (Ollama)"
	@echo "  make stage2            # download imagens (só util=true)"
	@echo "  make stage3-pass2      # normaliza nome + brand (Ollama)"
	@echo "  make stage4            # build dataset HF local"
	@echo "  make stage4-push       # push privado pro HF (precisa login)"
	@echo "  make stats             # status de todas as etapas"
	@echo "  make all               # 1 → 3p1 → 2 → 3p2 → 4 (sequencial)"
	@echo ""
	@echo "Variáveis: OLLAMA_MODEL=$(OLLAMA_MODEL)  HF_REPO=$(HF_REPO)"

all: stage1 stage3-pass1 stage2 stage3-pass2 stage4

stage1:
	python3 stage1_consolidate/consolidate.py

stage1-stats:
	python3 stage1_consolidate/consolidate.py --stats

stage3-pass1:
	OLLAMA_MODEL=$(OLLAMA_MODEL) python3 stage3_enrich/enrich_ollama.py pass1

stage3-pass2:
	OLLAMA_MODEL=$(OLLAMA_MODEL) python3 stage3_enrich/enrich_ollama.py pass2

stage3-stats:
	python3 stage3_enrich/enrich_ollama.py stats

stage2:
	python3 stage2_imgs/download_imgs.py --only-util --sleep 1.5

stage2-stats:
	python3 stage2_imgs/download_imgs.py stats

stage4:
	python3 stage4_pack/pack_and_push.py build

stage4-push:
	python3 stage4_pack/pack_and_push.py push --repo $(HF_REPO)

stats: stage1-stats stage3-stats stage2-stats
	@echo "" && python3 stage4_pack/pack_and_push.py stats || true

clean:
	@echo "Nada vai ser deletado automaticamente. Pra resetar uma etapa:"
	@echo "  rm stage1_consolidate/dataset_v0.parquet     # reset etapa 1"
	@echo "  rm stage3_enrich/pass1_classified.jsonl      # reset etapa 3 pass 1"
	@echo "  rm stage3_enrich/pass2_normalized.jsonl      # reset etapa 3 pass 2"
	@echo "  rm stage2_imgs/img_index.jsonl               # NÃO apaga imagens (~/dataset_imgs/)"
