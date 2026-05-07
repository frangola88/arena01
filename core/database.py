"""
Conexão SQLite thread-safe.
REGRA: sempre crie get_db() DENTRO da thread que vai usá-la.
O BackgroundTask do FastAPI roda em thread separada — crie a conexão lá dentro.
"""
import sqlite3
from datetime import datetime, date
from core.config import DB_PATH


# Adapters/converters explícitos: o adapter implícito do sqlite3 para datetime
# foi deprecado em Python 3.12 e será removido em 3.14.
def _adapt_datetime(val: datetime) -> str:
    return val.isoformat(sep=" ")


def _adapt_date(val: date) -> str:
    return val.isoformat()


def _convert_datetime(val: bytes) -> datetime:
    return datetime.fromisoformat(val.decode())


def _convert_date(val: bytes) -> date:
    return date.fromisoformat(val.decode())


sqlite3.register_adapter(datetime, _adapt_datetime)
sqlite3.register_adapter(date, _adapt_date)
# CURRENT_TIMESTAMP do SQLite vira string "YYYY-MM-DD HH:MM:SS"; PARSE_DECLTYPES
# casa pelo nome declarado da coluna (DATETIME). TIMESTAMP é incluído por hábito.
sqlite3.register_converter("datetime", _convert_datetime)
sqlite3.register_converter("DATETIME", _convert_datetime)
sqlite3.register_converter("timestamp", _convert_datetime)
sqlite3.register_converter("TIMESTAMP", _convert_datetime)
sqlite3.register_converter("date", _convert_date)
sqlite3.register_converter("DATE", _convert_date)


def get_db() -> sqlite3.Connection:
    conn = sqlite3.connect(
        str(DB_PATH),
        check_same_thread=False,
        detect_types=sqlite3.PARSE_DECLTYPES | sqlite3.PARSE_COLNAMES,
    )
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db() -> None:
    """Cria tabelas e popula dados iniciais. Idempotente."""
    conn = get_db()
    try:
        conn.executescript(SCHEMA_SQL)
        conn.executescript(SEED_SQL)
        # Migração leve: adiciona `progresso` se ainda não existir.
        for tabela in ("fotos_processadas", "videos_processados"):
            try:
                conn.execute(f"ALTER TABLE {tabela} ADD COLUMN progresso TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass  # coluna já existe
        conn.commit()
        print("[Database] Banco inicializado.")
    finally:
        conn.close()


SCHEMA_SQL = """
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
    icone TEXT DEFAULT '\U0001F4E6'
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

CREATE INDEX IF NOT EXISTS idx_objetos_localizacao    ON objetos(localizacao_id);
CREATE INDEX IF NOT EXISTS idx_objetos_categoria      ON objetos(categoria_id);
CREATE INDEX IF NOT EXISTS idx_objetos_nome           ON objetos(nome);
CREATE INDEX IF NOT EXISTS idx_objetos_palavras_chave ON objetos(palavras_chave);
"""

SEED_SQL = """
INSERT OR IGNORE INTO categorias (nome, grupo, icone) VALUES
    ('Ferramentas',   'Casa',        '\U0001F527'),
    ('Eletrônicos',   'Tecnologia',  '\U0001F4A1'),
    ('Roupas',        'Pessoal',     '\U0001F455'),
    ('Calçados',      'Pessoal',     '\U0001F45F'),
    ('Documentos',    'Pessoal',     '\U0001F4C4'),
    ('Livros',        'Educação',    '\U0001F4DA'),
    ('Brinquedos',    'Infantil',    '\U0001F9F8'),
    ('Cozinha',       'Casa',        '\U0001F373'),
    ('Limpeza',       'Casa',        '\U0001F9F9'),
    ('Decoração',     'Casa',        '\U0001F5BC️'),
    ('Esportes',      'Lazer',       '⚽'),
    ('Medicamentos',  'Saúde',       '\U0001F48A'),
    ('Papelaria',     'Escritório',  '✏️'),
    ('Cabos e Fios',  'Tecnologia',  '\U0001F50C'),
    ('Peças',         'Manutenção',  '⚙️'),
    ('Outros',        'Geral',       '\U0001F4E6');
"""
