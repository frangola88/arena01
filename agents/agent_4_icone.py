"""
Agente 4: Gerador de Imagem do Objeto

4 estratégias em cascata (para na primeira bem-sucedida):

  1. recorte_local   — redimensiona o recorte para 256x256 (melhor quando confiança >= 0.75)
  2. busca_web       — DuckDuckGo Images (gratuito, sem API key)
                       baixa imagem real do produto da internet
  3. claude_desenha  — Claude API gera JSON de formas geométricas, PIL executa
                       (produz ilustração reconhecível mesmo sem foto)
  4. placeholder_PIL — fundo colorido + emoji + nome (sempre funciona, sem rede)

IMPORTANTE:
  - objeto_id deve ser o cursor.lastrowid (INSERT já feito no banco)
  - Retorna (icone_path, icone_fonte) para o pipeline atualizar o banco
  - ddgs.images(query=...) é a assinatura correta do pacote 'ddgs' >= 9.0.0
"""
import io
from pathlib import Path
from PIL import Image, ImageDraw
from typing import Optional
from core.config import ICONES_DIR, ANTHROPIC_API_KEY, LIMIAR_CONFIANCA


CORES_GRUPO = {
    "Casa": (230, 245, 230), "Tecnologia": (225, 235, 250),
    "Pessoal": (250, 230, 245), "Educação": (255, 245, 220),
    "Infantil": (255, 235, 225), "Lazer": (230, 250, 250),
    "Saúde": (240, 255, 240), "Escritório": (245, 245, 255),
    "Manutenção": (245, 240, 225), "Geral": (240, 240, 240),
}


def _nome_util(nome: str) -> bool:
    genericos = {"objeto", "coisa", "item", "peca", "desconhecido", "multiplos", "varios"}
    return len(nome.strip()) > 3 and not any(g in nome.lower() for g in genericos)


# -- Estratégia 1: recorte local ----------------------------------------------

def _recorte_como_icone(recorte_path: str, caminho_saida: str) -> bool:
    try:
        img = Image.open(recorte_path).convert("RGB")
        img = img.resize((256, 256), Image.LANCZOS)
        img.save(caminho_saida, "PNG")
        return True
    except Exception as e:
        print(f"[Agente 4] Recorte falhou: {e}")
        return False


# -- Estratégia 2: busca web DuckDuckGo ---------------------------------------

def _buscar_imagem_web(nome: str, caminho_saida: str) -> bool:
    """
    Usa ddgs.images(query=...) — assinatura correta do pacote ddgs >= 9.0.0.
    Tenta até 5 URLs antes de desistir.
    """
    try:
        from ddgs import DDGS
        import requests
        with DDGS() as ddgs:
            resultados = ddgs.images(
                query=f"{nome} produto foto",
                max_results=5,
            )
        for item in resultados:
            try:
                resp = requests.get(
                    item["image"], timeout=8,
                    headers={"User-Agent": "Mozilla/5.0 CasaIQ/1.0"}
                )
                if resp.status_code == 200 and len(resp.content) > 2000:
                    img = Image.open(io.BytesIO(resp.content)).convert("RGB")
                    img = img.resize((256, 256), Image.LANCZOS)
                    img.save(caminho_saida, "PNG")
                    print(f"[Agente 4] Imagem web: {item['image'][:60]}...")
                    return True
            except Exception:
                continue
    except Exception as e:
        print(f"[Agente 4] Busca web falhou: {e}")
    return False


# -- Estratégia 3: Claude desenha (JSON -> PIL) -------------------------------

PROMPT_DESENHO = """Crie uma ilustracao simples do objeto: "{nome}"
Canvas 256x256 pixels, fundo branco.
Responda APENAS com JSON (nao inclua texto fora do JSON):
{{"fundo":[255,255,255],"formas":[
  {{"tipo":"retangulo","x1":60,"y1":80,"x2":196,"y2":170,"cor":[160,160,160],"espessura":3}},
  {{"tipo":"texto","x":128,"y":230,"texto":"{nome_curto}","cor":[60,60,60],"ancora":"mm"}}
]}}
Tipos: retangulo, elipse, linha, texto, poligono (com "pontos":[x1,y1,x2,y2,...]).
Use no maximo 15 formas. Represente o objeto de forma clara e reconhecivel."""


def _executar_desenho_PIL(instrucoes: dict, caminho_saida: str) -> bool:
    try:
        fundo = tuple(instrucoes.get("fundo", [255, 255, 255]))
        img   = Image.new("RGB", (256, 256), color=fundo)
        draw  = ImageDraw.Draw(img)
        for f in instrucoes.get("formas", []):
            tipo = f.get("tipo", "")
            cor  = tuple(f.get("cor", [0, 0, 0]))
            if tipo == "retangulo":
                draw.rectangle([f["x1"], f["y1"], f["x2"], f["y2"]],
                                outline=cor, width=f.get("espessura", 1))
            elif tipo == "elipse":
                bb = [f["x1"], f["y1"], f["x2"], f["y2"]]
                if f.get("preenchido"):
                    draw.ellipse(bb, fill=cor)
                else:
                    draw.ellipse(bb, outline=cor)
            elif tipo == "linha":
                draw.line([f["x1"], f["y1"], f["x2"], f["y2"]],
                           fill=cor, width=f.get("espessura", 1))
            elif tipo == "texto":
                draw.text((f["x"], f["y"]), str(f.get("texto", "")),
                           fill=cor, anchor=f.get("ancora", "la"))
            elif tipo == "poligono":
                pts = f.get("pontos", [])
                if len(pts) >= 6:
                    draw.polygon(pts, outline=cor)
        img.save(caminho_saida, "PNG")
        return True
    except Exception as e:
        print(f"[Agente 4] Executor PIL falhou: {e}")
        return False


def _claude_desenha(nome: str, caminho_saida: str) -> bool:
    if not ANTHROPIC_API_KEY:
        return False
    try:
        import anthropic
        from core.llm import extrair_json
        nome_curto = nome[:15]
        prompt = PROMPT_DESENHO.format(nome=nome, nome_curto=nome_curto)
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        resp = client.messages.create(
            model="claude-sonnet-4-20250514", max_tokens=800,
            messages=[{"role": "user", "content": prompt}]
        )
        instrucoes = extrair_json(resp.content[0].text)
        ok = _executar_desenho_PIL(instrucoes, caminho_saida)
        if ok:
            print(f"[Agente 4] Claude desenhou: '{nome}'")
        return ok
    except Exception as e:
        print(f"[Agente 4] Claude desenho falhou: {e}")
        return False


# -- Estratégia 4: placeholder PIL (fallback final) ---------------------------

def _placeholder_PIL(nome: str, icone_emoji: str, grupo: str, caminho_saida: str) -> bool:
    try:
        cor = CORES_GRUPO.get(grupo, (240, 240, 240))
        img  = Image.new("RGB", (256, 256), color=cor)
        draw = ImageDraw.Draw(img)
        try:
            draw.text((128, 100), icone_emoji, anchor="mm", fill=(60, 60, 60))
        except Exception:
            draw.rectangle([88, 60, 168, 140], fill=(180, 180, 180))
        texto = nome[:18] if len(nome) <= 18 else nome[:15] + "..."
        draw.text((128, 185), texto, anchor="mm", fill=(80, 80, 80))
        img.save(caminho_saida, "PNG")
        return True
    except Exception as e:
        print(f"[Agente 4] Placeholder falhou: {e}")
        return False


# -- Função principal ---------------------------------------------------------

def gerar_icone(recorte_path: str, nome: str, categoria_nome: str,
                icone_emoji: str, grupo: str, confianca: float,
                objeto_id: int) -> tuple[str, str]:
    """
    Tenta as 4 estratégias em ordem. Para na primeira bem-sucedida.
    Retorna (icone_path, icone_fonte).
    objeto_id DEVE ser o cursor.lastrowid — INSERT já feito antes desta chamada.
    """
    caminho = str(ICONES_DIR / f"objeto_{objeto_id:06d}.png")
    print(f"[Agente 4] Gerando ícone para '{nome}' (confiança={confianca:.2f})")

    # 1. Recorte local (alta qualidade)
    if confianca >= 0.75 and recorte_path:
        if _recorte_como_icone(recorte_path, caminho):
            return caminho, "recorte"

    # 2. Busca web
    if _nome_util(nome):
        if _buscar_imagem_web(nome, caminho):
            return caminho, "web"

    # 3. Recorte com baixa confiança (melhor que placeholder)
    if recorte_path and confianca >= 0.3:
        if _recorte_como_icone(recorte_path, caminho):
            return caminho, "recorte_baixa_qualidade"

    # 4. Claude desenha
    if _nome_util(nome):
        if _claude_desenha(nome, caminho):
            return caminho, "claude_desenho"

    # 5. Placeholder PIL (sempre funciona)
    _placeholder_PIL(nome, icone_emoji, grupo, caminho)
    return caminho, "placeholder"
