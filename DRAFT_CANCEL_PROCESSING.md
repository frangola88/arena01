# Rascunho: Cancelamento de Processamento

> **Status:** Rascunho adicional ao sistema de progresso
> **Propósito:** Permitir ao usuário parar processamento em andamento

---

## 1. BACKEND — Endpoint para cancelar

```python
# api/routes/fotos.py (RASCUNHO - adicionar)

@router.post("/fotos/{foto_id}/cancelar")
def cancelar_processamento_foto(foto_id: int):
    """
    Cancela o processamento de uma foto em andamento.
    
    Muda o status para "erro" e armazena mensagem "Cancelado pelo usuário".
    """
    conn = get_db()
    try:
        # Verificar se existe e está processando
        foto = conn.execute(
            "SELECT status FROM fotos_processadas WHERE id = ?", (foto_id,)
        ).fetchone()
        
        if not foto:
            raise HTTPException(status_code=404, detail="Foto não encontrada")
        
        if foto["status"] not in ["pendente", "processando"]:
            raise HTTPException(
                status_code=400, 
                detail=f"Não pode cancelar foto com status '{foto['status']}'"
            )
        
        # Marcar como cancelado
        conn.execute(
            """UPDATE fotos_processadas 
               SET status = ?, 
                   erro_mensagem = ?, 
                   concluido_em = CURRENT_TIMESTAMP
               WHERE id = ?""",
            ("erro", "Cancelado pelo usuário", foto_id)
        )
        conn.commit()
        
        return {"mensagem": "Processamento cancelado", "foto_id": foto_id}
    
    finally:
        conn.close()
```

---

## 2. FRONTEND — Botão Cancelar no Progress Component

```jsx
// web/src/components/ProcessingProgress.jsx (ATUALIZADO - RASCUNHO)

import React, { useState, useEffect } from 'react';
import './ProcessingProgress.css';

export function ProcessingProgress({ fotoId, onComplete, onCancel }) {
  const [status, setStatus] = useState(null);
  const [cancelando, setCancelando] = useState(false);

  useEffect(() => {
    if (!fotoId) return;

    // Poll a cada 500ms para atualizar progresso
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/fotos/${fotoId}/progresso`);
        const data = await res.json();
        setStatus(data);

        // Se concluído ou erro, parar polling
        if (data.etapa_atual === 'concluído' || 
            data.etapa_atual === 'erro' || 
            data.percentual === 100) {
          clearInterval(interval);
          if (onComplete) onComplete();
        }
      } catch (err) {
        console.error('Erro ao buscar progresso:', err);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [fotoId, onComplete]);

  const handleCancelar = async () => {
    if (!window.confirm('Tem certeza que quer cancelar o processamento?')) {
      return;
    }

    setCancelando(true);
    try {
      const res = await fetch(`/api/fotos/${fotoId}/cancelar`, {
        method: 'POST',
      });

      if (res.ok) {
        const data = await res.json();
        console.log(data.mensagem);
        
        // Atualizar status para mostrar erro/cancelado
        setStatus({
          ...status,
          etapa_atual: 'erro',
          erro_mensagem: 'Cancelado pelo usuário'
        });
        
        if (onCancel) onCancel();
      } else {
        alert('Erro ao cancelar: ' + res.statusText);
      }
    } catch (err) {
      console.error('Erro ao cancelar:', err);
      alert('Erro ao cancelar processamento');
    } finally {
      setCancelando(false);
    }
  };

  if (!status) {
    return <div className="progress-container">Iniciando...</div>;
  }

  const formatarTempo = (segundos) => {
    if (segundos === null) return '...';
    const m = Math.floor(segundos / 60);
    const s = Math.floor(segundos % 60);
    return `${m}m ${s}s`;
  };

  // Se foi cancelado, não mostrar mais o botão
  const podeCanc = status.etapa_atual !== 'erro' && status.etapa_atual !== 'concluído';

  return (
    <div className="progress-container">
      {/* Descrição do que está acontecendo */}
      <div className="progress-description">
        <div className="etapa-icon">
          {status.etapa_atual === 'segmentando' && '✂️'}
          {status.etapa_atual === 'analisando' && '🔍'}
          {status.etapa_atual === 'enriquecendo' && '✨'}
          {status.etapa_atual === 'finalizando' && '📦'}
          {status.etapa_atual === 'concluído' && '✅'}
          {status.etapa_atual === 'erro' && '❌'}
        </div>
        <div className="description-text">
          <p className="etapa-nome">
            {status.etapa_atual === 'segmentando' && 'Segmentando objetos'}
            {status.etapa_atual === 'analisando' && 'Analisando detalhes'}
            {status.etapa_atual === 'enriquecendo' && 'Enriquecendo dados'}
            {status.etapa_atual === 'finalizando' && 'Finalizando'}
            {status.etapa_atual === 'concluído' && 'Concluído!'}
            {status.etapa_atual === 'erro' && 'Erro no processamento'}
          </p>
          <p className="descricao-detalhe">
            {status.erro_mensagem || status.descricao_etapa}
          </p>
        </div>
      </div>

      {/* Barra de progresso */}
      <div className="progress-bar-container">
        <div className="progress-bar">
          <div 
            className="progress-fill" 
            style={{ width: `${status.percentual}%` }}
          />
        </div>
        <div className="progress-label">
          {status.percentual}% ({status.objetos_processados}/{status.objetos_totais} objetos)
        </div>
      </div>

      {/* Tempo */}
      <div className="progress-timing">
        <span className="tempo-decorrido">
          ⏱️ {formatarTempo(status.tempo_decorrido_s)}
        </span>
        {status.tempo_restante_s !== null && (
          <span className="tempo-restante">
            ⏳ ~{formatarTempo(status.tempo_restante_s)} faltando
          </span>
        )}
      </div>

      {/* === BOTÃO CANCELAR === */}
      {podeCanc && (
        <div className="progress-actions">
          <button 
            className="btn-cancelar" 
            onClick={handleCancelar}
            disabled={cancelando}
          >
            {cancelando ? '⏳ Cancelando...' : '⛔ Cancelar Processamento'}
          </button>
        </div>
      )}

      {status.erro_mensagem && (
        <div className="progress-error">
          ⚠️ {status.erro_mensagem}
        </div>
      )}
    </div>
  );
}
```

---

## 3. CSS para botão Cancelar

```css
/* web/src/components/ProcessingProgress.css (ATUALIZAR - RASCUNHO) */

.progress-actions {
  margin-top: 15px;
  display: flex;
  gap: 10px;
  justify-content: center;
}

.btn-cancelar {
  background: #e74c3c;
  color: white;
  border: none;
  padding: 10px 20px;
  border-radius: 6px;
  font-size: 0.95em;
  font-weight: bold;
  cursor: pointer;
  transition: all 0.3s ease;
  box-shadow: 0 2px 4px rgba(231, 76, 60, 0.3);
}

.btn-cancelar:hover:not(:disabled) {
  background: #c0392b;
  transform: translateY(-2px);
  box-shadow: 0 4px 8px rgba(231, 76, 60, 0.4);
}

.btn-cancelar:active:not(:disabled) {
  transform: translateY(0);
}

.btn-cancelar:disabled {
  background: #bdc3c7;
  cursor: not-allowed;
  opacity: 0.7;
}
```

---

## 4. Alternativa: Tecla ESC para cancelar (RASCUNHO)

```jsx
// Se quiser permitir ESC também:

useEffect(() => {
  const handleKeyPress = (e) => {
    if (e.key === 'Escape' && podeCanc && !cancelando) {
      handleCancelar();
    }
  };

  window.addEventListener('keydown', handleKeyPress);
  return () => window.removeEventListener('keydown', handleKeyPress);
}, [fotoId, cancelando]);
```

---

## 5. Fluxo visual esperado

**Antes de cancelar:**
```
✂️ Segmentando objetos
Recortando 12 de 45 objetos...
████████░░░░░░░░░░░░░░ 35%
⏱️ 8s  |  ⏳ ~16s faltando

[⛔ Cancelar Processamento]  ← Botão vermelho
```

**Depois de clicar Cancelar:**
```
❌ Erro no processamento
Cancelado pelo usuário
████████░░░░░░░░░░░░░░ 35%
⏱️ 8s

⚠️ Cancelado pelo usuário
```

---

## 📋 Checklist de implementação

- [ ] Adicionar endpoint POST `/api/fotos/{id}/cancelar`
- [ ] Atualizar componente React com botão
- [ ] Adicionar CSS para botão vermelho
- [ ] Opcional: suporte a ESC para cancelar
- [ ] Testar cancelo em diferentes etapas

---

*Rascunho criado em 2026-05-13*
*Pronto para enxertia quando aprovado*
