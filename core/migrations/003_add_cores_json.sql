-- Migração 003: adiciona cores_json na tabela objetos
-- cores_json: JSON array de {hex, nome, area_pct, parte} — paleta de cores
-- dominantes extraída pelo Claude, usada pelo GazetteerMatcher como guia.

ALTER TABLE objetos ADD COLUMN cores_json TEXT DEFAULT '[]';
