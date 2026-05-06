"""Rotas /modelos: lista modelos do Ollama por tipo e troca os ativos."""
import re
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import ollama
from core.runtime import get_vision_model, get_text_model, set_models

router = APIRouter()

# Heurística: nomes que sugerem capacidade de visão.
PADRAO_VISAO = re.compile(r"(vl|llava|cogvlm|moondream|bakllava|llama3\.2-vision)", re.IGNORECASE)


class TrocaModelosBody(BaseModel):
    vision_model: str | None = None
    text_model:   str | None = None


def _tipo(nome: str) -> str:
    return "vision" if PADRAO_VISAO.search(nome) else "text"


@router.get("/modelos")
def listar_modelos():
    """Lista modelos instalados no Ollama, separados por capacidade inferida."""
    try:
        resp = ollama.list()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama indisponível: {e}")

    nomes = []
    for m in resp.get("models", []):
        nome = m.get("name") or m.get("model") or ""
        if nome:
            nomes.append(nome)
    nomes.sort()

    visao = [n for n in nomes if _tipo(n) == "vision"]
    texto = [n for n in nomes if _tipo(n) == "text"]
    return {
        "vision_models": visao,
        "text_models":   texto,
        "atual": {
            "vision_model": get_vision_model(),
            "text_model":   get_text_model(),
        },
    }


@router.post("/modelos")
def trocar_modelos(body: TrocaModelosBody):
    """Troca o modelo ativo de visão e/ou texto. Persiste só em memória (até reiniciar)."""
    if not body.vision_model and not body.text_model:
        raise HTTPException(status_code=400, detail="Informe vision_model e/ou text_model")

    # Validar contra a lista do Ollama
    try:
        resp = ollama.list()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama indisponível: {e}")

    instalados = {(m.get("name") or m.get("model") or "") for m in resp.get("models", [])}
    for campo, valor in (("vision_model", body.vision_model),
                         ("text_model",   body.text_model)):
        if valor and valor not in instalados:
            raise HTTPException(status_code=400,
                                detail=f"Modelo '{valor}' não está instalado no Ollama")

    estado = set_models(vision_model=body.vision_model, text_model=body.text_model)
    return {"ok": True, "atual": estado}
