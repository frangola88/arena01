-- Migration 001: Initial schema for CasaIQ v3
-- Created: 2026-05-13
-- Description: Create all tables for initial setup

CREATE TABLE IF NOT EXISTS localizacoes (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    nome        TEXT NOT NULL,
    tipo        TEXT NOT NULL DEFAULT 'caixa',
    comodo      TEXT,
    descricao   TEXT,
    criado_em   DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS categorias (
    id    INTEGER PRIMARY KEY AUTOINCREMENT,
    nome  TEXT NOT NULL UNIQUE,
    grupo TEXT,
    icone TEXT DEFAULT '📦'
);

CREATE TABLE IF NOT EXISTS objetos (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    nome                  TEXT NOT NULL,
    descricao             TEXT,
    categoria_id          INTEGER REFERENCES categorias(id),
    localizacao_id        INTEGER REFERENCES localizacoes(id),
    cor                   TEXT,
    tamanho               TEXT,
    tamanho_estimado_cm   TEXT,
    peso_estimado_g       INTEGER,
    material              TEXT,
    estado                TEXT DEFAULT 'bom',
    funcao                TEXT,
    palavras_chave        TEXT DEFAULT '',
    foto_original_path    TEXT,
    recorte_path          TEXT,
    icone_path            TEXT,
    icone_fonte           TEXT DEFAULT '',
    confianca             REAL DEFAULT 1.0,
    modelo_visao          TEXT DEFAULT '',
    revisado_pelo_usuario INTEGER DEFAULT 0,
    criado_em             DATETIME DEFAULT CURRENT_TIMESTAMP,
    atualizado_em         DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TRIGGER IF NOT EXISTS trg_objetos_atualizado_em
AFTER UPDATE ON objetos FOR EACH ROW
BEGIN
    UPDATE objetos SET atualizado_em = CURRENT_TIMESTAMP WHERE id = OLD.id;
END;

CREATE TABLE IF NOT EXISTS fotos_processadas (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    caminho              TEXT NOT NULL,
    localizacao_id       INTEGER REFERENCES localizacoes(id),
    status               TEXT DEFAULT 'pendente',
    progresso            TEXT DEFAULT '',
    objetos_encontrados  INTEGER DEFAULT 0,
    erro_mensagem        TEXT,
    iniciado_em          DATETIME,
    concluido_em         DATETIME,
    criado_em            DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS videos_processados (
    id                   INTEGER PRIMARY KEY AUTOINCREMENT,
    caminho              TEXT NOT NULL,
    localizacao_id       INTEGER REFERENCES localizacoes(id),
    status               TEXT DEFAULT 'pendente',
    progresso            TEXT DEFAULT '',
    frames_extraidos     INTEGER DEFAULT 0,
    frames_processados   INTEGER DEFAULT 0,
    objetos_encontrados  INTEGER DEFAULT 0,
    erro_mensagem        TEXT,
    iniciado_em          DATETIME,
    concluido_em         DATETIME,
    criado_em            DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS historico_chat (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    pergunta  TEXT NOT NULL,
    resposta  TEXT NOT NULL,
    modelo    TEXT DEFAULT 'ollama',
    criado_em DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Índices para performance
CREATE INDEX IF NOT EXISTS idx_objetos_localizacao ON objetos(localizacao_id);
CREATE INDEX IF NOT EXISTS idx_objetos_categoria ON objetos(categoria_id);
CREATE INDEX IF NOT EXISTS idx_objetos_nome ON objetos(nome);
CREATE INDEX IF NOT EXISTS idx_objetos_palavras_chave ON objetos(palavras_chave);
CREATE INDEX IF NOT EXISTS idx_historico_criado ON historico_chat(criado_em);
CREATE INDEX IF NOT EXISTS idx_fotos_status ON fotos_processadas(status);
CREATE INDEX IF NOT EXISTS idx_videos_status ON videos_processados(status);
