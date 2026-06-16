# Rascunho: Sistema de Progresso em Tempo Real

> **Status:** Rascunho para review posterior
> **Propósito:** Evitar impressão de "travado" durante processamento
> **Implementação:** Backend + Frontend com WebSocket/polling

---

## 1. BACKEND — Endpoints de Status

### Modelo de Resposta (draft)

```python
# core/models.py (RASCUNHO)

from pydantic import BaseModel
from enum import Enum

class EtapaProcessamento(str, Enum):
    """Etapas do pipeline de processamento."""
    INICIANDO = "iniciando"
    SEGMENTANDO = "segmentando"      # Agent 1
    ANALISANDO = "analisando"        # Agent 2
    ENRIQUECENDO = "enriquecendo"    # Agent 3
    FINALIZANDO = "finalizando"
    CONCLUIDO = "concluído"
    ERRO = "erro"

class StatusProcessamento(BaseModel):
    """Status detalhado do processamento de uma foto/vídeo."""
    id: int                          # foto_processadas.id ou videos_processados.id
    tipo: str                        # "foto" ou "vídeo"
    nome_arquivo: str                # nome da foto/vídeo
    etapa_atual: EtapaProcessamento  # qual agente está rodando
    descricao_etapa: str             # "Recortando 12 de 45 objetos..."
    percentual: int                  # 0-100
    tempo_decorrido_s: float         # segundos desde início
    tempo_estimado_total_s: float    # ETA total (ou None se desconhecido)
    tempo_restante_s: float          # segundos faltantes (ou None)
    objetos_processados: int         # quantos objetos já foram processados
    objetos_totais: int              # total esperado
    erro_mensagem: str = None        # se houver erro
```

### Endpoint GET /api/fotos/{id}/progresso (RASCUNHO)

```python
# api/routes/fotos.py (FRAGMENTO)

from core.models import StatusProcessamento, EtapaProcessamento
from datetime import datetime

@router.get("/fotos/{foto_id}/progresso")
def obter_progresso_foto(foto_id: int):
    """
    Retorna status detalhado do processamento de uma foto.
    
    Chamado frequentemente pelo frontend (a cada 500ms) para atualizar barra de progresso.
    
    Response:
    {
      "id": 1,
      "tipo": "foto",
      "nome_arquivo": "IMG_001.jpg",
      "etapa_atual": "segmentando",
      "descricao_etapa": "Recortando 12 de 45 objetos detectados...",
      "percentual": 35,
      "tempo_decorrido_s": 8.2,
      "tempo_estimado_total_s": 24.5,
      "tempo_restante_s": 16.3,
      "objetos_processados": 12,
      "objetos_totais": 45
    }
    """
    conn = get_db()
    try:
        # Buscar registro de foto
        foto = conn.execute(
            "SELECT * FROM fotos_processadas WHERE id = ?", (foto_id,)
        ).fetchone()
        
        if not foto:
            raise HTTPException(status_code=404, detail="Foto não encontrada")
        
        # Parsing do campo "progresso" (JSON armazenado)
        progresso_json = json.loads(foto["progresso"] or "{}")
        
        # Calcular percentual e tempo restante
        inicio_em = datetime.fromisoformat(foto["iniciado_em"])
        tempo_decorrido = (datetime.utcnow() - inicio_em).total_seconds()
        
        # Lógica simples: se 12 de 45 objetos foram processados
        # e leva ~2s por objeto, estimado 90s total
        objetos_processados = progresso_json.get("objetos_processados", 0)
        objetos_totais = progresso_json.get("objetos_totais", 45)
        
        if objetos_processados > 0:
            tempo_por_objeto = tempo_decorrido / objetos_processados
            tempo_total_estimado = tempo_por_objeto * objetos_totais
            tempo_restante = max(0, tempo_total_estimado - tempo_decorrido)
            percentual = int((objetos_processados / objetos_totais) * 100)
        else:
            tempo_total_estimado = None
            tempo_restante = None
            percentual = 0
        
        return StatusProcessamento(
            id=foto_id,
            tipo="foto",
            nome_arquivo=foto["caminho"].split("/")[-1],
            etapa_atual=progresso_json.get("etapa", "iniciando"),
            descricao_etapa=progresso_json.get("descricao", "Processando..."),
            percentual=percentual,
            tempo_decorrido_s=tempo_decorrido,
            tempo_estimado_total_s=tempo_total_estimado,
            tempo_restante_s=tempo_restante,
            objetos_processados=objetos_processados,
            objetos_totais=objetos_totais
        )
    finally:
        conn.close()
```

### Campo "progresso" na tabela fotos_processadas (RASCUNHO)

```sql
-- core/migrations/003_add_progress_tracking.sql (RASCUNHO)

ALTER TABLE fotos_processadas ADD COLUMN progresso TEXT DEFAULT '{}';
-- Armazena JSON: {"etapa": "segmentando", "descricao": "...", "objetos_processados": 12, "objetos_totais": 45}

ALTER TABLE videos_processados ADD COLUMN progresso TEXT DEFAULT '{}';
-- Idem para vídeos
```

---

## 2. AGENT UPDATES — Atualizar campo "progresso"

### Agent 1: Segmentador (RASCUNHO)

```python
# agents/agent_1_segmentador.py (FRAGMENTO)

def processar_foto_com_progresso(foto_path: str, foto_id: int, conn: sqlite3.Connection):
    """
    Versão do processador que atualiza progresso em tempo real.
    """
    from PIL import Image
    import json
    
    # Carregar imagem
    img = Image.open(foto_path)
    
    # Detectar objetos (visão)
    objetos = chamar_visao(img, prompt="detecte todos os objetos")
    total_objetos = len(objetos["deteccoes"])
    
    recortes = []
    for i, obj in enumerate(objetos["deteccoes"]):
        # Recortar
        recorte_path = _recortar(img, obj, foto_id, i)
        recortes.append(recorte_path)
        
        # === ATUALIZAR PROGRESSO ===
        progresso = {
            "etapa": "segmentando",
            "descricao": f"Recortando {i+1} de {total_objetos} objetos...",
            "objetos_processados": i + 1,
            "objetos_totais": total_objetos
        }
        conn.execute(
            "UPDATE fotos_processadas SET progresso = ? WHERE id = ?",
            (json.dumps(progresso), foto_id)
        )
        conn.commit()
    
    return recortes
```

### Agent 2: Analisador (RASCUNHO)

```python
# agents/agent_2_analisador.py (FRAGMENTO)

def analisar_com_progresso(recortes: list, foto_id: int, conn: sqlite3.Connection):
    """
    Analisa cada recorte e atualiza progresso.
    """
    import json
    
    resultados = []
    total = len(recortes)
    
    for i, recorte_path in enumerate(recortes):
        # Analisar este recorte
        resultado = extrair_json(chamar_visao(Image.open(recorte_path), prompt="..."))
        resultados.append(resultado)
        
        # === ATUALIZAR PROGRESSO ===
        progresso = {
            "etapa": "analisando",
            "descricao": f"Analisando {i+1} de {total} objetos (nome, cor, tamanho)...",
            "objetos_processados": i + 1,
            "objetos_totais": total
        }
        conn.execute(
            "UPDATE fotos_processadas SET progresso = ? WHERE id = ?",
            (json.dumps(progresso), foto_id)
        )
        conn.commit()
    
    return resultados
```

### Agent 3: Enriquecedor (RASCUNHO)

```python
# agents/agent_3_enriquecedor.py (FRAGMENTO)

def enriquecer_com_progresso(objetos: list, foto_id: int, conn: sqlite3.Connection):
    """
    Enriquece cada objeto com metadados e atualiza progresso.
    """
    import json
    
    enriquecidos = []
    total = len(objetos)
    
    for i, obj in enumerate(objetos):
        # Enriquecer
        enriched = _enriquecer_objeto(obj)
        enriquecidos.append(enriched)
        
        # === ATUALIZAR PROGRESSO ===
        progresso = {
            "etapa": "enriquecendo",
            "descricao": f"Adicionando metadados em {i+1} de {total} objetos...",
            "objetos_processados": i + 1,
            "objetos_totais": total
        }
        conn.execute(
            "UPDATE fotos_processadas SET progresso = ? WHERE id = ?",
            (json.dumps(progresso), foto_id)
        )
        conn.commit()
    
    return enriquecidos
```

---

## 3. FRONTEND — Componente React de Progresso

### ProgressBar Component (RASCUNHO)

```jsx
// web/src/components/ProcessingProgress.jsx (RASCUNHO)

import React, { useState, useEffect } from 'react';
import './ProcessingProgress.css';

export function ProcessingProgress({ fotoId, onComplete }) {
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!fotoId) return;

    // Poll a cada 500ms para atualizar progresso
    const interval = setInterval(async () => {
      try {
        const res = await fetch(`/api/fotos/${fotoId}/progresso`);
        const data = await res.json();
        setStatus(data);

        // Se concluído, parar polling
        if (data.etapa_atual === 'concluído' || data.percentual === 100) {
          clearInterval(interval);
          if (onComplete) onComplete();
        }
      } catch (err) {
        console.error('Erro ao buscar progresso:', err);
      }
    }, 500);

    return () => clearInterval(interval);
  }, [fotoId, onComplete]);

  if (!status) {
    return <div className="progress-container">Iniciando...</div>;
  }

  const formatarTempo = (segundos) => {
    if (segundos === null) return '...';
    const m = Math.floor(segundos / 60);
    const s = Math.floor(segundos % 60);
    return `${m}m ${s}s`;
  };

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
          <p className="descricao-detalhe">{status.descricao_etapa}</p>
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

      {status.erro_mensagem && (
        <div className="progress-error">
          ⚠️ {status.erro_mensagem}
        </div>
      )}
    </div>
  );
}
```

### CSS para o Componente (RASCUNHO)

```css
/* web/src/components/ProcessingProgress.css (RASCUNHO) */

.progress-container {
  background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
  border-radius: 8px;
  padding: 20px;
  margin: 20px 0;
  box-shadow: 0 4px 6px rgba(0,0,0,0.1);
}

.progress-description {
  display: flex;
  align-items: center;
  gap: 15px;
  margin-bottom: 15px;
}

.etapa-icon {
  font-size: 2em;
  min-width: 40px;
  text-align: center;
}

.description-text {
  flex: 1;
}

.etapa-nome {
  margin: 0;
  font-weight: bold;
  font-size: 1.1em;
  color: #2c3e50;
}

.descricao-detalhe {
  margin: 5px 0 0 0;
  color: #555;
  font-size: 0.95em;
}

.progress-bar-container {
  margin: 15px 0;
}

.progress-bar {
  background: #ecf0f1;
  border-radius: 10px;
  height: 25px;
  overflow: hidden;
  border: 2px solid #3498db;
}

.progress-fill {
  height: 100%;
  background: linear-gradient(90deg, #3498db, #2ecc71);
  transition: width 0.3s ease;
  display: flex;
  align-items: center;
  justify-content: flex-end;
  padding-right: 8px;
  color: white;
  font-weight: bold;
  font-size: 0.9em;
}

.progress-label {
  text-align: center;
  font-size: 0.9em;
  color: #555;
  margin-top: 5px;
}

.progress-timing {
  display: flex;
  gap: 20px;
  justify-content: center;
  font-size: 0.95em;
  color: #7f8c8d;
  margin-top: 10px;
}

.tempo-decorrido, .tempo-restante {
  font-weight: 500;
}

.progress-error {
  background: #ffe6e6;
  border-left: 4px solid #e74c3c;
  padding: 10px;
  margin-top: 10px;
  border-radius: 4px;
  color: #c0392b;
}
```

### Hook para usar o componente (RASCUNHO)

```jsx
// web/src/hooks/useProcessingProgress.js (RASCUNHO)

import { useState, useCallback } from 'react';

export function useProcessingProgress(fotoId) {
  const [isProcessing, setIsProcessing] = useState(false);

  const startProcessing = useCallback(async () => {
    setIsProcessing(true);
    
    try {
      const res = await fetch(`/api/fotos/${fotoId}/progresso`);
      const data = await res.json();
      
      // Se já concluído, parar
      if (data.etapa_atual === 'concluído') {
        setIsProcessing(false);
      }
    } catch (err) {
      console.error('Erro:', err);
      setIsProcessing(false);
    }
  }, [fotoId]);

  return { isProcessing, startProcessing };
}
```

---

## 4. Integração na página de upload (RASCUNHO)

```jsx
// web/src/pages/ProcessarFotos.jsx (FRAGMENTO - onde usar)

import { ProcessingProgress } from '../components/ProcessingProgress';

export function ProcessarFotos() {
  const [fotoId, setFotoId] = useState(null);
  const [processando, setProcessando] = useState(false);

  const handleUpload = async (file) => {
    // Upload e obter ID
    const res = await fetch('/api/fotos/ingerir', { ... });
    const foto = await res.json();
    setFotoId(foto.id);
    setProcessando(true);
  };

  return (
    <div>
      {/* Upload UI */}
      <input type="file" onChange={(e) => handleUpload(e.target.files[0])} />

      {/* === COMPONENTE DE PROGRESSO === */}
      {processando && fotoId && (
        <ProcessingProgress 
          fotoId={fotoId}
          onComplete={() => {
            setProcessando(false);
            // Recarregar inventário, etc
          }}
        />
      )}
    </div>
  );
}
```

---

## 5. Melhorias futuras (RASCUNHO)

- [ ] WebSocket em vez de polling (menos latência, menos requisições)
- [ ] Animação na barra de progresso (efeito "pulsante")
- [ ] Preview dos objetos sendo processados (mostrar thumbnails)
- [ ] Pausa/resume do processamento
- [ ] Histórico de processamentos anteriores
- [ ] Cancelamento de tarefa em progresso
- [ ] Persistência do status mesmo se usuário recarregar página

---

## Notas de implementação

1. **Schema update**: Adicionar coluna `progresso TEXT` em `fotos_processadas` e `videos_processados`
2. **Agent updates**: Cada agente precisa atualizar o campo `progresso` após processar cada objeto
3. **Endpoint**: GET `/api/fotos/{id}/progresso` calcula ETA baseado em tempo/objeto
4. **Frontend**: Component React polling a cada 500ms, mostra ícone + descrição + % + tempo
5. **Banco de dados**: Usar JSON para armazenar estado em progresso

---

*Rascunho criado em 2026-05-13*
*Pronto para enxertia nos arquivos quando aprovado*
