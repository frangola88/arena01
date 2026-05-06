"""Dataclasses do domínio CasaIQ."""
from dataclasses import dataclass
from typing import Optional
from datetime import datetime


@dataclass
class Objeto:
    id: Optional[int]              = None
    nome: str                      = ""
    descricao: str                 = ""
    categoria_id: Optional[int]    = None
    localizacao_id: Optional[int]  = None
    cor: str                       = ""
    tamanho: str                   = ""
    tamanho_estimado_cm: str       = ""
    peso_estimado_g: Optional[int] = None
    material: str                  = ""
    estado: str                    = "bom"
    funcao: str                    = ""
    palavras_chave: str            = ""   # formato: |token1|token2|
    foto_original_path: str        = ""
    recorte_path: str              = ""
    icone_path: str                = ""
    icone_fonte: str               = ""   # 'recorte'|'web'|'claude_desenho'|'placeholder'
    confianca: float               = 1.0
    modelo_visao: str              = ""   # 'ollama'|'claude_api'
    revisado_pelo_usuario: int     = 0
    criado_em: Optional[datetime]  = None
    atualizado_em: Optional[datetime] = None


@dataclass
class Localizacao:
    id: Optional[int]             = None
    nome: str                     = ""
    tipo: str                     = "caixa"
    comodo: str                   = ""
    descricao: str                = ""
    criado_em: Optional[datetime] = None


@dataclass
class Categoria:
    id: Optional[int] = None
    nome: str         = ""
    grupo: str        = ""
    icone: str        = "\U0001F4E6"


@dataclass
class FotoProcessada:
    id: Optional[int]              = None
    caminho: str                   = ""
    localizacao_id: Optional[int]  = None
    status: str                    = "pendente"
    objetos_encontrados: int       = 0
    erro_mensagem: Optional[str]   = None
    iniciado_em: Optional[datetime]  = None
    concluido_em: Optional[datetime] = None
    criado_em: Optional[datetime]    = None
