-- Migration 002: Add performance indices (if not already created by 001)
-- Created: 2026-05-13
-- Description: Ensure all indices exist for common queries

-- Already created in 001, but listed here for documentation
-- If running this separately (e.g., on existing DB), these are safe (IF NOT EXISTS)

CREATE INDEX IF NOT EXISTS idx_objetos_localizacao ON objetos(localizacao_id);
CREATE INDEX IF NOT EXISTS idx_objetos_categoria ON objetos(categoria_id);
CREATE INDEX IF NOT EXISTS idx_historico_criado ON historico_chat(criado_em);
CREATE INDEX IF NOT EXISTS idx_fotos_status ON fotos_processadas(status);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos_processados(status);
