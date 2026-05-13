"""
Testes unitários para agent_2_analisador — análise visual.

Foco: lógica de seleção de melhor resultado por confiança.
"""
from unittest.mock import patch
import pytest

from agents.agent_2_analisador import analisar_objeto


@pytest.fixture
def recorte_teste(tmp_path):
    """Cria um arquivo de recorte de teste."""
    recorte = tmp_path / "recorte.jpg"
    recorte.write_text("fake image")
    return str(recorte)


def test_analisar_retorna_resultado_com_confianca(recorte_teste):
    """analisar_objeto() retorna análise com confiança."""
    resposta = '{"nome": "alicate", "descricao": "ferramenta", "cor": "preto", "tamanho": "medio", "tamanho_estimado_cm": "20", "peso_estimado_g": 150, "material": "aco", "estado": "bom", "funcao": "cortar", "palavras_chave": ["metal"], "confianca": 0.95}'
    
    with patch("agents.agent_2_analisador.chamar_visao") as mock_visao:
        mock_visao.return_value = (resposta, "ollama")
        with patch("agents.agent_2_analisador.extrair_json") as mock_json:
            mock_json.return_value = {
                "nome": "alicate",
                "descricao": "ferramenta",
                "cor": "preto",
                "tamanho": "medio",
                "tamanho_estimado_cm": "20",
                "peso_estimado_g": 150,
                "material": "aco",
                "estado": "bom",
                "funcao": "cortar",
                "palavras_chave": ["metal"],
                "confianca": 0.95
            }
            with patch("agents.agent_2_analisador.deve_usar_claude_visao", return_value=False):
                resultado = analisar_objeto(recorte_teste, "alicate")
                
                assert resultado["nome"] == "alicate"
                assert resultado["confianca"] == 0.95
                assert resultado["_modelo"] == "ollama"


def test_analisar_palavra_chave_string_convertida(recorte_teste):
    """Palavras_chave como string é convertida para lista."""
    resposta = '{"nome": "martelo", "descricao": "ferramenta", "cor": "marrom", "tamanho": "medio", "tamanho_estimado_cm": "30", "peso_estimado_g": 500, "material": "madeira", "estado": "bom", "funcao": "bater", "palavras_chave": "ferramenta,carpintaria", "confianca": 0.90}'
    
    with patch("agents.agent_2_analisador.chamar_visao") as mock_visao:
        mock_visao.return_value = (resposta, "ollama")
        with patch("agents.agent_2_analisador.extrair_json") as mock_json:
            mock_json.return_value = {
                "nome": "martelo",
                "descricao": "ferramenta",
                "cor": "marrom",
                "tamanho": "medio",
                "tamanho_estimado_cm": "30",
                "peso_estimado_g": 500,
                "material": "madeira",
                "estado": "bom",
                "funcao": "bater",
                "palavras_chave": "ferramenta,carpintaria",
                "confianca": 0.90
            }
            with patch("agents.agent_2_analisador.deve_usar_claude_visao", return_value=False):
                resultado = analisar_objeto(recorte_teste, "martelo")
                
                assert isinstance(resultado["palavras_chave"], list)
                assert resultado["palavras_chave"] == ["ferramenta", "carpintaria"]


def test_analisar_retorna_maior_confianca(recorte_teste):
    """Com múltiplas análises, retorna a de maior confiança."""
    resposta1 = '{"nome": "obj1", "descricao": "desc", "cor": "cor", "tamanho": "p", "tamanho_estimado_cm": "1", "peso_estimado_g": 1, "material": "m", "estado": "bom", "funcao": "f", "palavras_chave": ["p1"], "confianca": 0.5}'
    resposta2 = '{"nome": "obj2", "descricao": "desc", "cor": "cor", "tamanho": "p", "tamanho_estimado_cm": "1", "peso_estimado_g": 1, "material": "m", "estado": "bom", "funcao": "f", "palavras_chave": ["p2"], "confianca": 0.9}'
    
    with patch("agents.agent_2_analisador.chamar_visao") as mock_visao:
        mock_visao.side_effect = [(resposta1, "ollama"), (resposta2, "claude")]
        with patch("agents.agent_2_analisador.extrair_json") as mock_json:
            mock_json.side_effect = [
                {"nome": "obj1", "descricao": "desc", "cor": "cor", "tamanho": "p", "tamanho_estimado_cm": "1", "peso_estimado_g": 1, "material": "m", "estado": "bom", "funcao": "f", "palavras_chave": ["p1"], "confianca": 0.5},
                {"nome": "obj2", "descricao": "desc", "cor": "cor", "tamanho": "p", "tamanho_estimado_cm": "1", "peso_estimado_g": 1, "material": "m", "estado": "bom", "funcao": "f", "palavras_chave": ["p2"], "confianca": 0.9}
            ]
            with patch("agents.agent_2_analisador.deve_usar_claude_visao", return_value=True):
                resultado = analisar_objeto(recorte_teste, "obj")
                
                assert resultado["nome"] == "obj2"
                assert resultado["confianca"] == 0.9


def test_analisar_erro_retorna_padrao(recorte_teste):
    """Se análise falha, retorna dados padrão."""
    with patch("agents.agent_2_analisador.chamar_visao") as mock_visao:
        mock_visao.side_effect = Exception("Ollama indisponível")
        with patch("agents.agent_2_analisador.extrair_json"):
            resultado = analisar_objeto(recorte_teste, "teste")
            
            assert resultado["nome"] == "teste"
            assert resultado["descricao"] == ""
            assert resultado["confianca"] == 0.0
            assert resultado["_modelo"] == "erro"
