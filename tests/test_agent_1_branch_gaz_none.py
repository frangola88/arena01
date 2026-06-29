"""
Teste: branch gaz is None em segmentar_foto → _gate_borda NÃO é chamado.

DECISÃO DE GRANULARIDADE (documentada conforme mandato):
    Optamos por exercitar segmentar_foto (orquestração real) com _gate_borda
    espionada (monkeypatch), em vez de testar a menor unidade isolada. Motivo:
    a condição `if gaz is not None and bbox_norm is not None:` (L250 em
    agent_1_segmentador.py) é a guarda que protege o gate. Testá-la via
    orquestração garante que nenhuma refatoração futura quebre o isolamento
    (gate não roda quando gaz=None) sem que o teste detecte.

    Custo de mock: analisar_cena, analisar_foto_completa, detectar_objetos,
    _dimensoes_imagem, verificar_cena, score_qualidade, refinar_crop,
    gerar_icone_anotado, GazetteerMatcher.get_instance e RECORTES_DIR.
    Todos substituídos por stubs mínimos — sem I/O, sem IA, sem rede.

REGRESSÃO TRAVADA:
    "O gate W(y) só roda quando há refinamento DINOv2 (gaz is not None)"
    — se alguém mover _gate_borda para fora do bloco `if gaz is not None`,
    este teste falha imediatamente.
"""
import sys
import types

import numpy as np
import pytest

# ── Mock de cv2 (mesmo padrão de test_agent_1_gate_borda.py) ──────────────────
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cena_minima():
    """AnaliseCena mínima válida, compatível com verificar_cena e score_qualidade."""
    from core.analise_cena import AnaliseCena
    sup = np.full((10, 10), 0.1, dtype=np.float32)
    bg  = np.zeros((10, 10), dtype=bool)
    return AnaliseCena(
        n_objetos_estimado=1,
        complexidade="simples",
        picos=[],
        bordas_ativas={"top": False, "bottom": False, "left": False, "right": False},
        superficie=sup,
        obj_mask=~bg,
        bg_mask=bg,
        shape_blocos=(10, 10),
        shape_original=(160, 160),
        tempo_s=0.0,
    )


def _objeto_aceitavel():
    """Dict de objeto mínimo aceito por _nome_eh_aceitavel e pelo loop principal."""
    return {
        "nome": "tesoura",
        "bbox_normalizada": {"x1": 0.2, "y1": 0.2, "x2": 0.5, "y2": 0.5},
        "cores_dominantes": [],
        "centroide_normalizado": {"cx": 0.35, "cy": 0.35},
        "confianca": 0.9,
    }


# ── Testes ────────────────────────────────────────────────────────────────────

class TestSegmentarFotoGazNone:
    """
    segmentar_foto com GazetteerMatcher.get_instance() → None:
    o gate _gate_borda deve permanecer intocado (0 chamadas).
    """

    def _patch_all(self, monkeypatch, tmp_path):
        """Instala todos os mocks necessários para segmentar_foto rodar sem I/O."""
        cena = _cena_minima()

        # ── imports lazy do módulo (agent_1 já deve estar em sys.modules por causa
        #    do mock de cv2 acima; se não estiver, importamos agora) ─────────────
        import agents.agent_1_segmentador as mod

        # Garante que RECORTES_DIR aponte para tmp_path (sem criar arquivos reais)
        monkeypatch.setattr(mod, "RECORTES_DIR", tmp_path)

        # analisar_cena → retorna cena mínima sem abrir arquivos
        monkeypatch.setattr(mod, "analisar_cena", lambda *a, **kw: cena)

        # analisar_foto_completa → análise mínima com um objeto aceitável
        monkeypatch.setattr(
            mod, "analisar_foto_completa",
            lambda *a, **kw: {"objetos": [_objeto_aceitavel()]},
        )

        # detectar_objetos → zero contornos CV (não precisamos do fallback)
        monkeypatch.setattr(mod, "detectar_objetos", lambda *a, **kw: [])

        # _dimensoes_imagem → sem PIL/EXIF
        monkeypatch.setattr(mod, "_dimensoes_imagem", lambda *a, **kw: (200, 200))

        # GazetteerMatcher.get_instance → None (este é o branch que testamos)
        from core.gazetteer import GazetteerMatcher
        monkeypatch.setattr(GazetteerMatcher, "get_instance", staticmethod(lambda: None))

        # verificar_cena → retorno mínimo válido (sem divergência)
        monkeypatch.setattr(
            mod, "verificar_cena",
            lambda *a, **kw: {
                "contagem": {"veredito": "ok", "ratio": 1.0, "n_claude": 1, "n_estimado": 1},
                "centroide_em_bg": [],
                "n_suspeitos_bg": 0,
            },
        )

        # score_qualidade → sem flags
        monkeypatch.setattr(
            mod, "score_qualidade",
            lambda *a, **kw: {"score": 1.0, "flags": []},
        )

        # refinar_crop → (None, "S2_bbox") sem criar arquivo
        monkeypatch.setattr(
            mod, "refinar_crop",
            lambda *a, **kw: (None, "S2_bbox"),
        )

        # gerar_icone_anotado → noop (não gera arquivo)
        monkeypatch.setattr(mod, "gerar_icone_anotado", lambda *a, **kw: None)

        return mod

    def test_gate_borda_nao_chamado_quando_gaz_none(self, monkeypatch, tmp_path):
        """
        ASSERT PRINCIPAL: com gaz=None, _gate_borda deve ser chamado 0 vezes.

        Estratégia: substituímos agents.agent_1_segmentador._gate_borda por um
        wrapper spy que registra as chamadas. Após segmentar_foto retornar,
        verificamos que o spy não foi acionado.
        """
        mod = self._patch_all(monkeypatch, tmp_path)

        # Spy em _gate_borda: registra chamadas sem mudar o comportamento
        gate_chamadas = []
        gate_original = mod._gate_borda

        def _gate_spy(*args, **kwargs):
            gate_chamadas.append((args, kwargs))
            return gate_original(*args, **kwargs)

        monkeypatch.setattr(mod, "_gate_borda", _gate_spy)

        # Executa a orquestração real
        resultado = mod.segmentar_foto("foto_fake.jpg", foto_id=1)

        # Verificação principal
        assert len(gate_chamadas) == 0, (
            f"_gate_borda foi chamado {len(gate_chamadas)} vez(es) com gaz=None. "
            "O gate deve rodar SOMENTE dentro do bloco `if gaz is not None and bbox_norm is not None`. "
            "Verifique se ele foi movido para fora dessa guarda em agent_1_segmentador.py."
        )

    def test_segmentar_foto_retorna_objeto_quando_gaz_none(self, monkeypatch, tmp_path):
        """
        Corretude básica: mesmo com gaz=None, segmentar_foto deve retornar
        o objeto esperado (bbox original do Claude é preservada).
        """
        mod = self._patch_all(monkeypatch, tmp_path)

        resultado = mod.segmentar_foto("foto_fake.jpg", foto_id=2)

        assert isinstance(resultado, list), "segmentar_foto deve retornar lista"
        assert len(resultado) == 1, f"Esperado 1 objeto, got {len(resultado)}"
        assert resultado[0]["nome"] == "tesoura"

    def test_bbox_original_preservada_quando_gaz_none(self, monkeypatch, tmp_path):
        """
        Com gaz=None, a bbox_normalizada do objeto NÃO é refinada nem rejeitada.
        O objeto no resultado deve carregar a bbox original do Claude.

        Verifica que `bbox_norm = obj.get("bbox_normalizada")` (linha do loop)
        permanece como veio de analisar_foto_completa, sem ser alterada pelo gate.
        """
        mod = self._patch_all(monkeypatch, tmp_path)

        # Captura o obj passado para refinar_crop para verificar a bbox
        obj_recebido = {}

        def _refinar_crop_spy(caminho_foto, obj, saida, numero, total_objetos, nome):
            obj_recebido.update(obj)
            return (None, "S2_bbox")

        monkeypatch.setattr(mod, "refinar_crop", _refinar_crop_spy)

        mod.segmentar_foto("foto_fake.jpg", foto_id=3)

        bbox_esperada = _objeto_aceitavel()["bbox_normalizada"]
        bbox_real = obj_recebido.get("bbox_normalizada")
        assert bbox_real == bbox_esperada, (
            f"bbox_normalizada deveria ser a original do Claude {bbox_esperada}, "
            f"mas chegou ao refinar_crop como {bbox_real}. "
            "O gate pode estar alterando a bbox mesmo com gaz=None."
        )
