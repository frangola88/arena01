"""
Estimador de peso — calcula peso aproximado baseado em:
- Categoria do objeto
- Material
- Tamanho estimado
- Densidade típica
"""
import logging

_log = logging.getLogger("casaiq.estimador_peso")


# Peso médio (em gramas) por categoria — valores típicos para itens domésticos
PESOS_CATEGORIA = {
    "Ferramentas": 200,
    "Utensílios Cozinha": 150,
    "Eletrônicos": 300,
    "Decoração": 100,
    "Limpeza": 80,
    "Têxtil": 50,
    "Plástico": 40,
    "Metal": 150,
    "Vidro": 200,
    "Cerâmica": 180,
    "Papelaria": 20,
    "Higiene": 30,
    "Outros": 100,
}

# Multiplicadores por tamanho
MULTIPLICADORES_TAMANHO = {
    "pequeno": 0.5,
    "medio": 1.0,
    "grande": 2.5,
}

# Multiplicadores por material
MULTIPLICADORES_MATERIAL = {
    "metal": 1.2,
    "aço": 1.3,
    "ferro": 1.4,
    "vidro": 1.1,
    "cerâmica": 1.0,
    "plástico": 0.6,
    "borracha": 0.7,
    "madeira": 0.8,
    "tecido": 0.3,
    "papel": 0.1,
    "alumínio": 0.9,
    "cobre": 1.5,
}


def estimar_peso(
    categoria_nome: str = None,
    tamanho: str = None,
    material: str = None,
    tamanho_estimado_cm: str = None,
) -> int | None:
    """
    Estima peso em gramas baseado em categoria, tamanho e material.
    
    Args:
        categoria_nome: nome da categoria (ex: "Ferramentas")
        tamanho: "pequeno", "medio" ou "grande"
        material: material principal (ex: "metal", "plástico")
        tamanho_estimado_cm: string com dimensões (ex: "10x5x2" ou "diameter 5cm")
    
    Returns:
        Peso estimado em gramas (int) ou None se não for possível estimar
    """
    if not categoria_nome:
        return None
    
    # 1. Peso base da categoria
    peso_base = PESOS_CATEGORIA.get(categoria_nome, PESOS_CATEGORIA["Outros"])
    
    # 2. Aplicar multiplicador de tamanho
    if tamanho:
        tam_norm = tamanho.strip().lower()
        multiplicador_tam = MULTIPLICADORES_TAMANHO.get(tam_norm, 1.0)
        peso_base *= multiplicador_tam
    
    # 3. Aplicar multiplicador de material
    if material:
        mat_norm = material.strip().lower()
        # Tenta match exato primeiro, depois substring
        if mat_norm in MULTIPLICADORES_MATERIAL:
            peso_base *= MULTIPLICADORES_MATERIAL[mat_norm]
        else:
            # Tenta encontrar substring
            for mat_chave, mult in MULTIPLICADORES_MATERIAL.items():
                if mat_chave in mat_norm or mat_norm in mat_chave:
                    peso_base *= mult
                    break
    
    # 4. Ajuste fino baseado em dimensões estimadas (opcional)
    if tamanho_estimado_cm:
        ajuste = _ajuste_por_dimensoes(tamanho_estimado_cm)
        peso_base *= ajuste
    
    return int(round(peso_base))


def _ajuste_por_dimensoes(tamanho_str: str) -> float:
    """
    Extrai dimensões de uma string e calcula ajuste volumétrico.
    Exemplos: "10x5x2", "diameter 5cm", "5cm x 3cm"
    
    Retorna multiplicador (0.5 a 2.0) baseado no volume relativo.
    """
    import re
    
    if not tamanho_str:
        return 1.0
    
    # Extrair números
    numeros = re.findall(r"\d+(?:\.\d+)?", tamanho_str)
    if not numeros:
        return 1.0
    
    numeros = [float(n) for n in numeros]
    
    # Se há 3 dimensões, calcular volume
    if len(numeros) >= 3:
        volume = numeros[0] * numeros[1] * numeros[2]
        # Volume referência: 10x5x2 = 100 cm³
        volume_ref = 100.0
        multiplicador = max(0.5, min(2.0, volume / volume_ref))
        return multiplicador
    
    # Se há 2 dimensões (área)
    elif len(numeros) == 2:
        area = numeros[0] * numeros[1]
        area_ref = 50.0  # 10x5
        multiplicador = max(0.6, min(1.8, area / area_ref))
        return multiplicador
    
    # Se há 1 dimensão (comprimento/diâmetro)
    elif len(numeros) == 1:
        tamanho_valor = numeros[0]
        tamanho_ref = 10.0  # referência: 10cm
        multiplicador = max(0.4, min(2.0, tamanho_valor / tamanho_ref))
        return multiplicador
    
    return 1.0


def formatar_peso(peso_g: int | None) -> str:
    """
    Formata peso para exibição legível.
    
    Args:
        peso_g: peso em gramas
    
    Returns:
        String formatada (ex: "250 g", "1.2 kg", "—")
    """
    if peso_g is None or peso_g == 0:
        return "—"
    
    if peso_g >= 1000:
        kg = peso_g / 1000
        return f"{kg:.1f} kg"
    else:
        return f"{peso_g} g"
