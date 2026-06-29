-- Migração 004: adiciona geometria_vetor na tabela objetos
-- geometria_vetor: GeoJSON do polígono vetorial (raster→vector) derivado da
-- superfície de object-ness (AnaliseCena.superficie) pelo vetorizador_raster.
-- Campo TEXT vazio por padrão; preenchido aditivamente pelo pipeline.

ALTER TABLE objetos ADD COLUMN geometria_vetor TEXT DEFAULT '';
