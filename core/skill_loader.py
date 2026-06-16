"""
Carregador de skills — lê arquivo markdown, aplica substituições e devolve o prompt.

Uma "skill" no CasaIQ é um arquivo .md em core/skills/ contendo:
- Instruções estruturadas para o LLM
- Schema esperado do output
- Vocabulário controlado
- Exemplos few-shot
- Placeholders {NOME_DA_VARIAVEL} que são substituídos em runtime

Vantagem: a inteligência do sistema fica DOCUMENTADA e VERSIONADA em um arquivo
único, não espalhada em strings de prompt no código. Iteração é diff-friendly.
"""
import logging
from pathlib import Path

_log = logging.getLogger("casaiq.skill_loader")

SKILLS_DIR = Path(__file__).parent / "skills"


class SkillNaoEncontrada(Exception):
    pass


def carregar_skill(nome_skill: str, variaveis: dict | None = None) -> str:
    """
    Carrega uma skill por nome e aplica substituições.
    
    Args:
        nome_skill: nome do arquivo sem extensão. Ex: "inventario_visao"
        variaveis: dict com placeholders → valores. Ex: {"CATEGORIAS_DISPONIVEIS": "..."}
    
    Returns:
        String do prompt renderizado.
    
    Raises:
        SkillNaoEncontrada: se o arquivo não existe.
    """
    caminho = SKILLS_DIR / f"{nome_skill}.md"
    if not caminho.exists():
        raise SkillNaoEncontrada(f"Skill '{nome_skill}' não encontrada em {caminho}")
    
    template = caminho.read_text(encoding="utf-8")
    
    if variaveis:
        for chave, valor in variaveis.items():
            placeholder = "{" + chave + "}"
            template = template.replace(placeholder, str(valor))
    
    # Detectar placeholders não-substituídos (defensivo)
    import re
    nao_substituidos = re.findall(r"\{[A-Z_]{3,}\}", template)
    if nao_substituidos:
        _log.warning("skill_placeholders_nao_substituidos", extra={
            "skill": nome_skill,
            "placeholders": list(set(nao_substituidos)),
        })
    
    _log.info("skill_carregada", extra={
        "skill": nome_skill,
        "tamanho_chars": len(template),
        "variaveis_substituidas": list((variaveis or {}).keys()),
    })
    return template


def listar_skills_disponiveis() -> list[str]:
    """Retorna lista de skills (sem .md) disponíveis."""
    if not SKILLS_DIR.exists():
        return []
    return sorted([p.stem for p in SKILLS_DIR.glob("*.md")])
