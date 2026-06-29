"""
Testes unitários diretos do gate W(y) — _gate_borda em agents/agent_1_segmentador.py.

Cobre a decisão isolada:
  (a) W baixo  → mantém bbox refinada
  (b) W alto   → reverte para bbox original
  (c) Cena degradada / bbox=None → não quebra, passthrough (retorna refinada)

Mecanismo de import: injeta types.ModuleType("cv2") em sys.modules ANTES de
importar _gate_borda, para não depender de OpenCV instalado (mesmo padrão de
test_agent_1_segmentador_unit.py).
"""
import logging
import sys
import types

import numpy as np
import pytest

# ── Injeta cv2 mock antes de qualquer import do módulo agent_1 ────────────────
if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

from agents.agent_1_segmentador import _gate_borda       # noqa: E402
from core.analise_cena import AnaliseCena                 # noqa: E402
from core.verificacoes_cruzadas import (                  # noqa: E402
    LIM_BORDA, FAIXA_BORDA, w_borda_bbox,
)


# ── Helper: fábrica de AnaliseCena mínima (mesmo padrão de
#            test_verificacoes_cruzadas.py) ─────────────────────────────────────

def _cena(bg_mask: np.ndarray, n_estimado: int = 4,
          superficie: np.ndarray | None = None) -> AnaliseCena:
    """Monta uma AnaliseCena mínima com bg_mask (e opcionalmente superficie)."""
    Hb, Wb = bg_mask.shape
    if superficie is None:
        superficie = np.where(bg_mask, 0.1, 0.8).astype(np.float32)
    return AnaliseCena(
        n_objetos_estimado=n_estimado,
        complexidade="media",
        picos=[],
        bordas_ativas={"top": False, "bottom": False, "left": False, "right": False},
        superficie=superficie,
        obj_mask=~bg_mask,
        bg_mask=bg_mask,
        shape_blocos=(Hb, Wb),
        shape_original=(Hb * 16, Wb * 16),
        tempo_s=0.0,
    )


# ── Bboxes de teste ────────────────────────────────────────────────────────────

# Bbox "refinada" (interna, resultado do DINOv2)
BBOX_REFINADA = {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8}

# Bbox "original" do Claude — diferente da refinada para asserts de identidade
BBOX_ORIGINAL = {"x1": 0.1, "y1": 0.1, "x2": 0.9, "y2": 0.9}


# ── Caso (a): W baixo → mantém bbox REFINADA ─────────────────────────────────

class TestGateBordaWBaixo:
    """
    Superficie com objeto isolado no centro (fundo ao redor).
    A faixa além da bbox deve ter pouco sinal de objeto → wb["max"] <= 0.5
    → _gate_borda deve retornar a bbox REFINADA.
    """

    def _cena_objeto_no_centro(self):
        # Grade 10×10: quase tudo fundo (0.05), objeto só em [4:6, 4:6]
        sup = np.full((10, 10), 0.05, dtype=np.float32)
        sup[4:6, 4:6] = 0.9
        return _cena(np.zeros((10, 10), dtype=bool), superficie=sup)

    def test_retorna_bbox_refinada_quando_w_baixo(self):
        cena = self._cena_objeto_no_centro()
        resultado = _gate_borda(BBOX_REFINADA, BBOX_ORIGINAL, cena)
        # Deve retornar a bbox REFINADA (o gate NÃO rejeita)
        assert resultado == BBOX_REFINADA

    def test_retorna_refinada_nao_original(self):
        """Confirma que o retorno é a refinada, não a original."""
        cena = self._cena_objeto_no_centro()
        resultado = _gate_borda(BBOX_REFINADA, BBOX_ORIGINAL, cena)
        assert resultado != BBOX_ORIGINAL

    def test_kwargs_foto_id_e_nome_nao_quebram(self):
        """foto_id e nome passados como kwargs não devem causar erro."""
        cena = self._cena_objeto_no_centro()
        resultado = _gate_borda(
            BBOX_REFINADA, BBOX_ORIGINAL, cena,
            foto_id=42, nome="chave de fenda"
        )
        assert resultado == BBOX_REFINADA


# ── Caso (b): W alto → reverte para bbox ORIGINAL ────────────────────────────

class TestGateBordaWAlto:
    """
    Superficie toda "objeto" (0.9): qualquer bbox interna terá overflow alto
    na faixa além da bbox → wb["max"] > 0.5 → _gate_borda deve retornar
    a bbox ORIGINAL (fallback para bbox do Claude).

    Alinhado com test_w_borda_detecta_objeto_continuando_alem_da_bbox
    em test_verificacoes_cruzadas.py (valida que w_borda_bbox realmente
    retorna max > 0.5 nesse cenário).
    """

    def _cena_overflow(self):
        sup = np.full((10, 10), 0.9, dtype=np.float32)
        return _cena(np.zeros((10, 10), dtype=bool), superficie=sup)

    def test_retorna_bbox_original_quando_w_alto(self):
        cena = self._cena_overflow()
        # bbox_refinada interna, portanto tem overflow nas faixas externas
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}
        resultado = _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena)
        # O gate REJEITA a refinada e devolve a original
        assert resultado == BBOX_ORIGINAL

    def test_retorna_original_nao_refinada(self):
        """Confirma que o retorno é a original, não a refinada."""
        cena = self._cena_overflow()
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}
        resultado = _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena)
        assert resultado != bbox_refinada

    def test_com_foto_id_e_nome_loga_sem_quebrar(self):
        """Caminho de rejeição deve funcionar com todos os kwargs."""
        cena = self._cena_overflow()
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}
        resultado = _gate_borda(
            bbox_refinada, BBOX_ORIGINAL, cena,
            foto_id=7, nome="tesoura"
        )
        assert resultado == BBOX_ORIGINAL


# ── Caso (c): Borda / degradado / None → não quebra, passthrough ─────────────

class TestGateBordaDegradado:
    """
    Cenários de borda: cena degradada (superficie 1×1 → w_borda_bbox retorna
    max=0.0) e bbox_refinada=None.
    Nenhum desses casos deve levantar exceção.
    Quando a cena não tem sinal, o gate não rejeita (retorna a refinada / None).
    """

    def _cena_degradada(self):
        # superficie 1×1 → w_borda_bbox retorna all-zero (sem sinal)
        return _cena(np.ones((1, 1), dtype=bool), n_estimado=0)

    def test_cena_degradada_nao_quebra_e_retorna_refinada(self):
        cena = self._cena_degradada()
        resultado = _gate_borda(BBOX_REFINADA, BBOX_ORIGINAL, cena)
        # Sem sinal → wb["max"] == 0.0 → não rejeita → retorna refinada
        assert resultado == BBOX_REFINADA

    def test_bbox_refinada_none_nao_quebra(self):
        """bbox_refinada=None deve ser passada transparentemente (sem exceção)."""
        cena = self._cena_degradada()
        resultado = _gate_borda(None, BBOX_ORIGINAL, cena)
        # w_borda_bbox(None, ...) retorna all-zero → gate não rejeita → passthrough
        assert resultado is None

    def test_bbox_refinada_none_cena_valida_nao_quebra(self):
        """Com cena válida (superficie alta), bbox_refinada=None ainda não quebra."""
        sup = np.full((10, 10), 0.9, dtype=np.float32)
        cena = _cena(np.zeros((10, 10), dtype=bool), superficie=sup)
        # w_borda_bbox(None, ...) retorna {cortado: [], max: 0.0, ...}
        resultado = _gate_borda(None, BBOX_ORIGINAL, cena)
        assert resultado is None

    def test_ambos_none_nao_quebra(self):
        """Ambas as bboxes None — função não deve levantar exceção."""
        cena = self._cena_degradada()
        resultado = _gate_borda(None, None, cena)
        assert resultado is None

    def test_cena_degradada_retorna_refinada_nao_original(self):
        """Sem sinal, o gate NÃO usa o fallback — mantém refinada."""
        cena = self._cena_degradada()
        resultado = _gate_borda(BBOX_REFINADA, BBOX_ORIGINAL, cena)
        assert resultado != BBOX_ORIGINAL


# ── ITEM 1: Teste de log via caplog (caminho de REJEIÇÃO, W alto) ─────────────

class TestGateBordaLogRejeicao:
    """
    Captura o LogRecord emitido pelo caminho de REJEIÇÃO do gate W(y) e
    verifica que a mensagem e todos os campos extra estão corretos.

    Usa pytest caplog (handler adicionado no root logger) para interceptar
    WARNINGs do logger "casaiq.agent_1". Funciona porque os loggers casaiq.*
    propagam para o root (sem propagate=False), então caplog os alcança.

    Cenário: superficie toda 0.9 (overflow alto) → _gate_borda REJEITA a
    bbox refinada e emite o WARNING com nome, foto_id, cortado_em, w_max.
    """

    def _cena_overflow(self):
        sup = np.full((10, 10), 0.9, dtype=np.float32)
        return _cena(np.zeros((10, 10), dtype=bool), superficie=sup)

    def test_rejeicao_emite_warning_com_mensagem_correta(self, caplog):
        """O caminho de rejeição emite exatamente um WARNING com a chave esperada."""
        cena = self._cena_overflow()
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}

        with caplog.at_level(logging.WARNING, logger="casaiq.agent_1"):
            _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena, foto_id=7, nome="tesoura")

        # Filtrar só os records do logger correto com a mensagem esperada
        records = [
            r for r in caplog.records
            if r.name == "casaiq.agent_1"
            and r.msg == "objeto_cortado_rejeitando_bbox_refinada"
        ]
        assert len(records) == 1, (
            f"Esperado exatamente 1 record WARNING 'objeto_cortado_rejeitando_bbox_refinada', "
            f"encontrado {len(records)}"
        )
        record = records[0]
        assert record.levelno == logging.WARNING

    def test_rejeicao_extra_nome(self, caplog):
        """Extra 'nome' no LogRecord deve bater com o argumento passado."""
        cena = self._cena_overflow()
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}

        with caplog.at_level(logging.WARNING, logger="casaiq.agent_1"):
            _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena, foto_id=7, nome="tesoura")

        record = next(
            r for r in caplog.records
            if r.name == "casaiq.agent_1"
            and r.msg == "objeto_cortado_rejeitando_bbox_refinada"
        )
        assert getattr(record, "nome", None) == "tesoura"

    def test_rejeicao_extra_foto_id(self, caplog):
        """Extra 'foto_id' no LogRecord deve bater com o argumento passado."""
        cena = self._cena_overflow()
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}

        with caplog.at_level(logging.WARNING, logger="casaiq.agent_1"):
            _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena, foto_id=7, nome="tesoura")

        record = next(
            r for r in caplog.records
            if r.name == "casaiq.agent_1"
            and r.msg == "objeto_cortado_rejeitando_bbox_refinada"
        )
        assert getattr(record, "foto_id", None) == 7

    def test_rejeicao_extra_cortado_em(self, caplog):
        """Extra 'cortado_em' deve ser lista não-vazia de lados válidos."""
        cena = self._cena_overflow()
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}
        lados_validos = {"top", "bottom", "left", "right"}

        with caplog.at_level(logging.WARNING, logger="casaiq.agent_1"):
            _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena, foto_id=7, nome="tesoura")

        record = next(
            r for r in caplog.records
            if r.name == "casaiq.agent_1"
            and r.msg == "objeto_cortado_rejeitando_bbox_refinada"
        )
        cortado_em = getattr(record, "cortado_em", None)
        assert cortado_em is not None, "Extra 'cortado_em' ausente no LogRecord"
        assert isinstance(cortado_em, list), f"'cortado_em' deve ser lista, got {type(cortado_em)}"
        assert len(cortado_em) > 0, "'cortado_em' deve ser lista não-vazia"
        assert set(cortado_em).issubset(lados_validos), (
            f"'cortado_em' contém lados inválidos: {set(cortado_em) - lados_validos}"
        )

    def test_rejeicao_extra_w_max(self, caplog):
        """Extra 'w_max' deve ser round(max, 3) do w_borda_bbox recalculado — sem hardcode."""
        cena = self._cena_overflow()
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}

        # Recalcula o esperado usando a mesma função e constantes que _gate_borda usa
        wb_esperado = w_borda_bbox(bbox_refinada, cena, faixa=FAIXA_BORDA, th=LIM_BORDA)
        w_max_esperado = round(wb_esperado["max"], 3)

        with caplog.at_level(logging.WARNING, logger="casaiq.agent_1"):
            _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena, foto_id=7, nome="tesoura")

        record = next(
            r for r in caplog.records
            if r.name == "casaiq.agent_1"
            and r.msg == "objeto_cortado_rejeitando_bbox_refinada"
        )
        w_max_log = getattr(record, "w_max", None)
        assert w_max_log is not None, "Extra 'w_max' ausente no LogRecord"
        assert w_max_log == w_max_esperado, (
            f"w_max no log ({w_max_log}) != round(wb['max'], 3) esperado ({w_max_esperado})"
        )


# ── ITEM 4 (T4): Teste anti-drift — constantes LIM_BORDA e FAIXA_BORDA ───────

class TestGateBordaConstantesAntiDrift:
    """
    Trava a relação entre _gate_borda e as constantes LIM_BORDA / FAIXA_BORDA.

    Detecta drift futuro se alguém reintroduzir literais divergentes ou
    alterar os valores efetivos das constantes sem refletir no gate.

    (a) Valores efetivos: LIM_BORDA == 0.5 e FAIXA_BORDA == 0.04.
    (b) Decisão do gate é consistente com LIM_BORDA:
        - Superficie cujo overflow cai ACIMA de LIM_BORDA → rejeita (retorna BBOX_ORIGINAL).
        - Superficie cujo overflow cai ABAIXO de LIM_BORDA → aceita (retorna bbox_refinada).
    (c) O w_max logado bate com round(w_borda_bbox(..., faixa=FAIXA_BORDA)[\"max\"], 3).
    """

    def test_lim_borda_valor_efetivo(self):
        """LIM_BORDA deve permanecer 0.5 (trava o valor efetivo)."""
        assert LIM_BORDA == 0.5, (
            f"LIM_BORDA mudou para {LIM_BORDA}! "
            "Se foi intencional, atualize este teste E documente o impacto no gate."
        )

    def test_faixa_borda_valor_efetivo(self):
        """FAIXA_BORDA deve permanecer 0.04 (trava o valor efetivo)."""
        assert FAIXA_BORDA == 0.04, (
            f"FAIXA_BORDA mudou para {FAIXA_BORDA}! "
            "Se foi intencional, atualize este teste E documente o impacto no gate."
        )

    def _cena_com_overflow(self, valor_superficie: float):
        """Cena com superficie uniforme — controla o overflow exatamente."""
        sup = np.full((10, 10), valor_superficie, dtype=np.float32)
        return _cena(np.zeros((10, 10), dtype=bool), superficie=sup)

    def test_decisao_rejeicao_consistente_com_lim_borda(self):
        """
        Cenário com overflow ACIMA de LIM_BORDA: gate deve rejeitar a bbox refinada
        e retornar BBOX_ORIGINAL. Verifica que a decisão usa LIM_BORDA, não um literal.
        """
        # Superficie 0.9 garante overflow >> LIM_BORDA (0.5) em qualquer bbox interna
        cena = self._cena_com_overflow(0.9)
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}

        # Confirma que o sinal de fato excede LIM_BORDA com FAIXA_BORDA
        wb = w_borda_bbox(bbox_refinada, cena, faixa=FAIXA_BORDA, th=LIM_BORDA)
        assert wb["max"] > LIM_BORDA, (
            f"Pré-condição falhou: wb['max']={wb['max']} <= LIM_BORDA={LIM_BORDA}. "
            "O cenário não reproduz overflow acima do limiar."
        )

        resultado = _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena)
        assert resultado == BBOX_ORIGINAL, (
            f"Gate deveria rejeitar (overflow={wb['max']:.3f} > LIM_BORDA={LIM_BORDA}), "
            f"mas retornou {resultado}"
        )

    def test_decisao_aceitacao_consistente_com_lim_borda(self):
        """
        Cenário com overflow ABAIXO de LIM_BORDA: gate deve aceitar a bbox refinada.
        Verifica consistência com LIM_BORDA no caminho de aceitação.
        """
        # Superficie 0.05 garante overflow << LIM_BORDA (0.5) — objeto isolado no centro
        cena = self._cena_com_overflow(0.05)
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}

        # Confirma que o sinal fica abaixo de LIM_BORDA
        wb = w_borda_bbox(bbox_refinada, cena, faixa=FAIXA_BORDA, th=LIM_BORDA)
        assert wb["max"] < LIM_BORDA, (
            f"Pré-condição falhou: wb['max']={wb['max']} >= LIM_BORDA={LIM_BORDA}. "
            "O cenário não reproduz overflow abaixo do limiar."
        )

        resultado = _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena)
        assert resultado == bbox_refinada, (
            f"Gate deveria aceitar (overflow={wb['max']:.3f} < LIM_BORDA={LIM_BORDA}), "
            f"mas retornou {resultado}"
        )

    def test_w_max_logado_consistente_com_faixa_borda(self, caplog):
        """
        O campo w_max no LogRecord deve ser round(w_borda_bbox(..., faixa=FAIXA_BORDA)['max'], 3).
        Trava que o gate usa FAIXA_BORDA (não um literal 0.04 divergente) para calcular w_max.
        """
        cena = self._cena_com_overflow(0.9)
        bbox_refinada = {"x1": 0.3, "y1": 0.3, "x2": 0.6, "y2": 0.6}

        wb_esperado = w_borda_bbox(bbox_refinada, cena, faixa=FAIXA_BORDA, th=LIM_BORDA)
        w_max_esperado = round(wb_esperado["max"], 3)

        with caplog.at_level(logging.WARNING, logger="casaiq.agent_1"):
            _gate_borda(bbox_refinada, BBOX_ORIGINAL, cena, foto_id=99, nome="drift_test")

        record = next(
            (r for r in caplog.records
             if r.name == "casaiq.agent_1"
             and r.msg == "objeto_cortado_rejeitando_bbox_refinada"),
            None,
        )
        assert record is not None, "WARNING esperado não foi emitido"
        w_max_log = getattr(record, "w_max", None)
        assert w_max_log == w_max_esperado, (
            f"w_max no log ({w_max_log}) diverge de round(wb['max'], 3)={w_max_esperado}. "
            "Possível drift: o gate pode estar usando FAIXA_BORDA diferente."
        )
