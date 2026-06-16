"""
Módulo de deduplicação — detecta objetos duplicados usando similarity matching.
Evita criar múltiplos registros do mesmo objeto em diferentes fotos.
"""
import logging
from difflib import SequenceMatcher
from core.database import get_db

_log = logging.getLogger("casaiq.deduplicacao")


# Palavras-modificadoras que devem ser removidas para comparar nomes "puros"
# Ex: "chave de fenda vermelha" → "chave de fenda" para fins de comparação
MODIFICADORES = {
    # Cores
    "vermelho", "vermelha", "preto", "preta", "branco", "branca",
    "azul", "verde", "amarelo", "amarela", "laranja", "rosa", "cinza",
    "marrom", "roxo", "roxa", "bege", "dourado", "dourada", "prateado",
    # Tamanhos
    "pequeno", "pequena", "medio", "media", "grande", "mini",
    # Materiais quando não-essenciais
    "metalico", "metalica", "plastico", "plastica",
    # Outros adjetivos genéricos
    "novo", "nova", "velho", "velha", "usado", "usada",
    "bicolor", "colorido", "colorida", "ergonomico", "ergonomica",
    "transparente", "translucido",
    # Junções
    "com", "de", "do", "da", "e", "para",
}


def normalizar_texto(texto: str) -> str:
    """Remove acentos e converte para minúsculas."""
    import unicodedata
    if not texto:
        return ""
    nfd = unicodedata.normalize("NFD", texto.lower())
    return "".join(c for c in nfd if unicodedata.category(c) != "Mn")


def extrair_nucleo_nome(nome: str) -> str:
    """
    Extrai o "núcleo" do nome removendo modificadores (cor, tamanho, etc).
    Ex: "chave de fenda phillips com cabo vermelho" → "chave fenda phillips cabo"
    """
    if not nome:
        return ""
    n = normalizar_texto(nome)
    palavras = n.split()
    nucleo = [p for p in palavras if p not in MODIFICADORES and len(p) > 1]
    return " ".join(nucleo)


def corrigir_mojibake(texto: str) -> str:
    """
    Corrige mojibake comum (encoding bagunçado UTF-8 → Latin-1 → UTF-8).
    Casos típicos:
      "rÃ©gua" → "régua"
      "rÃ£gua" → "rãgua" → corrige para "régua" via dicionário
    """
    if not texto:
        return texto
    # Tentar reverter UTF-8 lido como Latin-1
    try:
        corrigido = texto.encode("latin-1").decode("utf-8")
        # Só aceita se o resultado tem menos caracteres "estranhos"
        chars_estranhos_antes = sum(1 for c in texto if ord(c) > 200 and c not in "áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ")
        chars_estranhos_depois = sum(1 for c in corrigido if ord(c) > 200 and c not in "áéíóúâêîôûãõçÁÉÍÓÚÂÊÎÔÛÃÕÇ")
        if chars_estranhos_depois < chars_estranhos_antes:
            return corrigido
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    # Fallback: substituições manuais de mojibake comum em PT-BR
    substituicoes = {
        "rãgua": "régua",
        "Ã©": "é",
        "Ã£": "ã",
        "Ã§": "ç",
        "Ã³": "ó",
        "Ã¡": "á",
        "Ã­": "í",
        "Ãª": "ê",
    }
    for ruim, bom in substituicoes.items():
        texto = texto.replace(ruim, bom)
    return texto


def limpar_nome_objeto(nome: str) -> str:
    """
    Limpa o nome de um objeto antes de salvar no banco:
    - Corrige mojibake
    - Remove cores e modificadores ("com cabo vermelho", "phillips", etc.)
    - Mantém apenas o núcleo legível
    
    Ex: "chave de fenda phillips com cabo vermelho e azul"
        → "chave de fenda"
    
    Ex: "régua de costura em madeira"
        → "régua de costura"
    """
    if not nome:
        return nome
    
    # 1. Corrigir encoding
    n = corrigir_mojibake(nome).strip().lower()
    
    # 2. Cortar tudo após " com " (descritores) — "X com cabo Y" → "X"
    for separador in [" com ", " de cor ", " na cor ", " coloridos", " coloridas"]:
        idx = n.find(separador)
        if idx > 0:
            n = n[:idx].strip()
    
    # 3. Remover sufixos com cores ("vermelho", "azul" no fim)
    palavras = n.split()
    while palavras and normalizar_texto(palavras[-1]) in MODIFICADORES:
        palavras.pop()
    n = " ".join(palavras)
    
    # 4. Remover "phillips" se for genérico (sem prefixo "chave de fenda")
    # Não fazemos isso — pode ser legítimo. Apenas garantimos não duplicar.
    if n == "chave phillips":
        n = "chave de fenda"
    if n == "chave de fenda phillips":
        n = "chave de fenda"  # tratamos phillips como não-confirmado
    
    return n.strip() or nome  # fallback se ficou vazio


def calcular_similaridade_nome(nome1: str, nome2: str) -> float:
    """Calcula similaridade entre dois nomes (0.0 a 1.0)."""
    n1 = normalizar_texto(nome1 or "")
    n2 = normalizar_texto(nome2 or "")
    if not n1 or not n2:
        return 0.0
    return SequenceMatcher(None, n1, n2).ratio()


def objetos_sao_duplicados(
    obj_novo: dict,
    obj_existente: dict,
    limiar_similaridade: float = 0.70,
) -> bool:
    """
    Verifica se dois objetos são duplicados.
    
    Estratégia em camadas:
    1. Compara o NÚCLEO do nome (sem cores/tamanhos/modificadores)
       — se núcleos idênticos, é duplicado direto
    2. Senão, usa similaridade textual com limiar 0.70
    3. Confirma com categoria + localização (devem bater)
    
    Args:
        obj_novo: novo objeto a inserir
        obj_existente: objeto já no banco
        limiar_similaridade: 0.0-1.0 (padrão 0.70 = 70% similar)
    
    Returns:
        True se parece duplicado, False caso contrário
    """
    nome_novo = obj_novo.get("nome", "")
    nome_exist = obj_existente.get("nome", "")
    
    # 1. Comparar NÚCLEOS (mais robusto contra variações de cor/tamanho)
    nucleo_novo = extrair_nucleo_nome(nome_novo)
    nucleo_exist = extrair_nucleo_nome(nome_exist)
    
    nucleos_iguais = (nucleo_novo and nucleo_exist and nucleo_novo == nucleo_exist)
    
    # 2. Se núcleos diferem, calcular similaridade textual
    if not nucleos_iguais:
        sim_nome = calcular_similaridade_nome(nome_novo, nome_exist)
        # Comparar também similaridade dos núcleos
        sim_nucleo = calcular_similaridade_nome(nucleo_novo, nucleo_exist) if nucleo_novo and nucleo_exist else 0.0
        sim_max = max(sim_nome, sim_nucleo)
        
        if sim_max < limiar_similaridade:
            return False  # Nomes muito diferentes
    
    # 3. Categoria — deve ser idêntica (ou ambas None)
    cat_nova = obj_novo.get("categoria_id")
    cat_exist = obj_existente.get("categoria_id")
    if (cat_nova or cat_exist) and cat_nova != cat_exist:
        return False
    
    # 4. Localização — deve ser idêntica
    loc_nova = obj_novo.get("localizacao_id")
    loc_exist = obj_existente.get("localizacao_id")
    if (loc_nova or loc_exist) and loc_nova != loc_exist:
        return False
    
    # OBS: NÃO comparamos mais "tamanho" — o mesmo objeto fotografado de
    # ângulos diferentes pode receber estimativas de tamanho ligeiramente
    # diferentes do VLM (pequeno/medio), o que estava criando falsos negativos.
    
    _log.info("duplicado_detectado", extra={
        "nome_novo": nome_novo,
        "nome_existente": nome_exist,
        "nucleo_novo": nucleo_novo,
        "nucleo_existente": nucleo_exist,
        "nucleos_iguais": nucleos_iguais,
    })
    return True


def buscar_objeto_similar(
    obj_novo: dict,
    localizacao_id: int,
    categoria_id: int = None,
    limiar_similaridade: float = 0.70,
    excluir_foto_path: str | None = None,
) -> dict | None:
    """
    Busca no banco um objeto similar ao novo.

    Args:
        obj_novo: dicionário com 'nome', 'tamanho' do novo objeto
        localizacao_id: localização onde o objeto foi detectado
        categoria_id: categoria do novo objeto (opcional)
        limiar_similaridade: threshold para match (0.0-1.0)
        excluir_foto_path: ignora objetos da mesma foto (evita falsos positivos
            quando há múltiplos itens do mesmo tipo na cena, ex: 3 alicates)

    Returns:
        Objeto duplicado encontrado (dict) ou None
    """
    conn = get_db()
    try:
        # Buscar candidatos: mesma localização e categoria (ou próximos)
        query = """
            SELECT id, nome, categoria_id, localizacao_id, tamanho
            FROM objetos
            WHERE localizacao_id = ?
        """
        params = [localizacao_id]

        # Nunca deduplica contra objetos da foto que está sendo processada agora
        if excluir_foto_path:
            query += " AND (foto_original_path IS NULL OR foto_original_path != ?)"
            params.append(excluir_foto_path)

        if categoria_id:
            query += " AND categoria_id = ?"
            params.append(categoria_id)
        
        candidatos = conn.execute(query, params).fetchall()
        
        for cand in candidatos:
            obj_exist = {
                "nome": cand["nome"],
                "categoria_id": cand["categoria_id"],
                "localizacao_id": cand["localizacao_id"],
                "tamanho": cand["tamanho"],
                "id": cand["id"],
            }
            
            if objetos_sao_duplicados(obj_novo, obj_exist, limiar_similaridade):
                return obj_exist
        
        return None
    finally:
        conn.close()


def registrar_duplicado(objeto_novo_id: int, objeto_original_id: int) -> None:
    """
    Registra a relação de duplicação no banco (se existir tabela de duplicatas).
    Por enquanto, apenas loga.
    """
    _log.info("duplicado_registrado", extra={
        "novo_id": objeto_novo_id,
        "original_id": objeto_original_id,
    })
