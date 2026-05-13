"""
Testes unitários para agent_3_enriquecedor — enriquecimento de metadados.

Foco: formatação de palavras_chave e seleção de categoria.
"""
from unittest.mock import patch
import pytest

from agents.agent_3_enriquecedor import enriquecer_objeto, _formatar_palavras_chave


# ============================================================================
# Testes: _formatar_palavras_chave()
# ============================================================================


def test_formatar_palavras_chave_lista_simples():
    """Formata lista com separador |."""
    tokens = ["ferramenta", "metal", "corte"]
    
    resultado = _formatar_palavras_chave(tokens)
    
    assert resultado == "|ferramenta|metal|corte|"


def test_formatar_palavras_chave_com_espacos():
    """Divide strings por espaço."""
    tokens = ["alicate de bico", "ferramenta"]
    
    resultado = _formatar_palavras_chave(tokens)
    
    assert "|alicate|" in resultado
    assert "|bico|" in resultado
    assert "|ferramenta|" in resultado


def test_formatar_palavras_chave_vazio():
    """Lista vazia retorna string vazia."""
    tokens = []
    
    resultado = _formatar_palavras_chave(tokens)
    
    assert resultado == ""


def test_formatar_palavras_chave_remove_duplicatas():
    """Remove palavras duplicadas."""
    tokens = ["metal", "corte", "metal", "ferramenta"]
    
    resultado = _formatar_palavras_chave(tokens)
    
    assert resultado.count("|metal|") == 1


def test_formatar_palavras_chave_minusculo():
    """Converte para minúsculo."""
    tokens = ["FERRAMENTA", "Metal", "CORTE"]
    
    resultado = _formatar_palavras_chave(tokens)
    
    assert resultado == "|ferramenta|metal|corte|"


def test_formatar_palavras_chave_ignora_muito_curtos():
    """Ignora tokens com menos de 2 caracteres."""
    tokens = ["a", "ferramenta", "b", "metal"]
    
    resultado = _formatar_palavras_chave(tokens)
    
    assert resultado == "|ferramenta|metal|"


# ============================================================================
# Testes: enriquecer_objeto()
# ============================================================================


def test_enriquecer_sucesso_com_categoria():
    """Enriquecimento bem-sucedido encontra categoria."""
    analise = {
        "nome": "alicate de bico",
        "descricao": "Ferramenta para cortar",
        "funcao": "cortar fios",
        "palavras_chave": ["metal", "corte"],
        "confianca": 0.9
    }
    categorias = ["Ferramentas", "Eletrônicos", "Roupas", "Outros"]
    resposta = '{"categoria": "Ferramentas", "palavras_chave_extras": ["eletricista"], "nome_normalizado": "alicate bico"}'
    
    with patch("agents.agent_3_enriquecedor.chamar_texto") as mock_texto:
        mock_texto.return_value = (resposta, "ollama")
        with patch("agents.agent_3_enriquecedor.extrair_json") as mock_json:
            mock_json.return_value = {
                "categoria": "Ferramentas",
                "palavras_chave_extras": ["eletricista"],
                "nome_normalizado": "alicate bico"
            }
            
            resultado = enriquecer_objeto(analise, categorias)
            
            assert resultado["categoria_nome"] == "Ferramentas"
            assert resultado["nome"] == "alicate bico"
            assert "|metal|" in resultado["palavras_chave"]
            assert "|eletricista|" in resultado["palavras_chave"]


def test_enriquecer_preserva_copia():
    """Enriquecimento não modifica análise original."""
    analise_original = {
        "nome": "martelo",
        "descricao": "Ferramenta de impacto",
        "funcao": "bater",
        "palavras_chave": ["metal"],
        "confianca": 0.95
    }
    resposta = '{"categoria": "Ferramentas", "palavras_chave_extras": [], "nome_normalizado": "martelo"}'
    
    with patch("agents.agent_3_enriquecedor.chamar_texto") as mock_texto:
        mock_texto.return_value = (resposta, "ollama")
        with patch("agents.agent_3_enriquecedor.extrair_json") as mock_json:
            mock_json.return_value = {
                "categoria": "Ferramentas",
                "palavras_chave_extras": [],
                "nome_normalizado": "martelo"
            }
            
            resultado = enriquecer_objeto(analise_original, ["Ferramentas"])
            
            # Original não foi modificado
            assert analise_original["nome"] == "martelo"
            assert resultado["confianca"] == 0.95


def test_enriquecer_erro_fallback_outros():
    """Erro → fallback categoria Outros."""
    analise = {
        "nome": "objeto",
        "descricao": "desc",
        "funcao": "func",
        "palavras_chave": ["palavra1"],
        "confianca": 0.8
    }
    
    with patch("agents.agent_3_enriquecedor.chamar_texto") as mock_texto:
        mock_texto.side_effect = Exception("Ollama offline")
        with patch("agents.agent_3_enriquecedor.extrair_json"):
            resultado = enriquecer_objeto(analise, ["Ferramentas"])
            
            assert resultado["categoria_nome"] == "Outros"
            assert "|palavra1|" in resultado["palavras_chave"]


def test_enriquecer_combina_palavras_chave():
    """Combina palavras_chave originais com extras."""
    analise = {
        "nome": "parafuso",
        "descricao": "Parafuso de metal",
        "funcao": "fixar",
        "palavras_chave": ["metal", "pequeno"],
        "confianca": 0.8
    }
    resposta = '{"categoria": "Peças", "palavras_chave_extras": ["fixacao"], "nome_normalizado": "parafuso"}'
    
    with patch("agents.agent_3_enriquecedor.chamar_texto") as mock_texto:
        mock_texto.return_value = (resposta, "ollama")
        with patch("agents.agent_3_enriquecedor.extrair_json") as mock_json:
            mock_json.return_value = {
                "categoria": "Peças",
                "palavras_chave_extras": ["fixacao"],
                "nome_normalizado": "parafuso"
            }
            
            resultado = enriquecer_objeto(analise, ["Peças"])
            
            assert "|metal|" in resultado["palavras_chave"]
            assert "|pequeno|" in resultado["palavras_chave"]
            assert "|fixacao|" in resultado["palavras_chave"]


def test_enriquecer_nome_adicionado_palavras_chave():
    """Nome normalizado é adicionado às palavras_chave."""
    analise = {
        "nome": "chave inglesa",
        "descricao": "Chave",
        "funcao": "apertar",
        "palavras_chave": [],
        "confianca": 0.9
    }
    resposta = '{"categoria": "Ferramentas", "palavras_chave_extras": [], "nome_normalizado": "chave ajustavel"}'
    
    with patch("agents.agent_3_enriquecedor.chamar_texto") as mock_texto:
        mock_texto.return_value = (resposta, "ollama")
        with patch("agents.agent_3_enriquecedor.extrair_json") as mock_json:
            mock_json.return_value = {
                "categoria": "Ferramentas",
                "palavras_chave_extras": [],
                "nome_normalizado": "chave ajustavel"
            }
            
            resultado = enriquecer_objeto(analise, ["Ferramentas"])
            
            assert "|chave|" in resultado["palavras_chave"]
            assert "|ajustavel|" in resultado["palavras_chave"]
