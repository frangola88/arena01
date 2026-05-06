from pydantic import BaseModel
from typing import Optional, List

class LocalizacaoCreate(BaseModel):
    nome: str
    tipo: str = "caixa"
    comodo: Optional[str] = None
    descricao: Optional[str] = None

class ObjetoUpdate(BaseModel):
    nome: Optional[str] = None
    descricao: Optional[str] = None
    categoria_id: Optional[int] = None
    localizacao_id: Optional[int] = None
    cor: Optional[str] = None
    tamanho: Optional[str] = None
    material: Optional[str] = None
    estado: Optional[str] = None
    funcao: Optional[str] = None
    palavras_chave: Optional[str] = None

class ChatRequest(BaseModel):
    pergunta: str
