# Manual UI Testing Checklist — CasaIQ v3

> **Purpose:** Comprehensive manual testing guide for CasaIQ features. Run through these scenarios on the web interface before each release.
>
> **Duration:** ~45 minutes per full suite
>
> **Date Last Updated:** 2026-05-13

---

## Setup

- [ ] **Start server:** `python -m uvicorn main:app --reload`
- [ ] **Open browser:** `http://localhost:8000`
- [ ] **Check /api/modo endpoint:** Verify correct operational mode (Ollama/Claude/OpenRouter)
- [ ] **Verify logs:** Check `tail -f logs/casaiq.log` for startup messages

---

## Section 1: Navigation & Layout

### 1.1 Interface Load
- [ ] Page loads without console errors (F12 → Console tab)
- [ ] All major sections visible: Localizações, Processar Fotos, Processar Vídeos, Objetos, Chat
- [ ] Navigation between sections works (no page reload)
- [ ] Responsive layout on desktop (test window resize)

### 1.2 Mode Indicator
- [ ] `/api/modo` endpoint displays correct operational mode
- [ ] Mode badge visible in UI (if implemented)
- [ ] Switching modes via config changes mode badge correctly

---

## Section 2: Localizações (Locations)

### 2.1 Create Location
- [ ] Click "Nova Localização" / "Add Location"
- [ ] Form appears with fields: Nome, Tipo, Cômodo, Descrição
- [ ] Submit with valid data (e.g., "Cozinha", tipo="cômodo", cômodo="Pavimento Inferior")
- [ ] Location appears in list after submit
- [ ] Form clears after successful submission

### 2.2 List Locations
- [ ] All created locations display in table/list
- [ ] Locations show: id, nome, tipo, cômodo, data criação
- [ ] List updates in real-time when new location added

### 2.3 Edit Location
- [ ] Click edit icon on a location
- [ ] Form populates with current values
- [ ] Change a field (e.g., descrição)
- [ ] Submit changes
- [ ] Updated value appears in list immediately

### 2.4 Delete Location
- [ ] Click delete icon on a location
- [ ] Confirmation dialog appears (optional but recommended)
- [ ] Location removed from list after confirmation

---

## Section 3: Processar Fotos (Photo Processing)

### 3.1 Upload Single Photo
- [ ] Click "Selecionar Foto" / file input
- [ ] Select a .jpg or .png from disk
- [ ] Select target localization from dropdown
- [ ] Click "Enviar" / "Process"
- [ ] Progress indicator shows (if implemented)
- [ ] Status changes from "pendente" → "processando" → "concluído" or "erro"

### 3.2 Photo Processing Results
- [ ] After processing, check API response status is "concluído"
- [ ] `objetos_encontrados` shows correct count (should be > 0 for real photos)
- [ ] Photo appears in system (verify in /storage if visible)
- [ ] No server errors in logs (tail -f logs/casaiq.log)

### 3.3 Processing Error Handling
- [ ] Upload invalid file format (e.g., .txt)
- [ ] Verify error message displays to user
- [ ] Status shows "erro" with erro_mensagem populated
- [ ] App doesn't crash

### 3.4 Multiple Photos
- [ ] Upload 2-3 photos to same location sequentially
- [ ] All process independently
- [ ] Check `fotos_processadas` table has 3+ entries with distinct paths

---

## Section 4: Processar Vídeos (Video Processing)

### 4.1 Upload Single Video
- [ ] Click "Selecionar Vídeo" / file input
- [ ] Select a .mp4 or similar video from disk
- [ ] Select target localization
- [ ] Click "Enviar" / "Process"
- [ ] Status shows "processando"

### 4.2 Video Processing Progress
- [ ] Progress bar or counter shows frames being extracted
- [ ] Frame extraction completes (frames_extraidos > 0)
- [ ] Frame processing begins automatically
- [ ] Status eventually becomes "concluído"

### 4.3 Video Results
- [ ] `frames_extraidos` shows count (e.g., 15-30 frames for ~1min video)
- [ ] `frames_processados` equals `frames_extraidos` (all processed)
- [ ] `objetos_encontrados` > 0 (objects detected across video frames)
- [ ] No crash on long processing

### 4.4 Video Error Handling
- [ ] Upload corrupted video file
- [ ] Verify error message and status = "erro"
- [ ] Check logs for detailed error
- [ ] App recovers gracefully

---

## Section 5: Objetos (Objects/Inventory)

### 5.1 Objects List After Photo/Video
- [ ] After processing, objects appear in "Objetos" section
- [ ] Each object shows: nome, categoria, localizacao, cor, tamanho, estado
- [ ] Icons display correctly for categories (use icone field)
- [ ] List is searchable/filterable by nome (if implemented)

### 5.2 Object Details Modal
- [ ] Click on an object to open details
- [ ] Modal/panel shows all fields:
  - [ ] nome, descricao, categoria
  - [ ] localizacao, cor, tamanho, tamanho_estimado_cm
  - [ ] peso_estimado_g, material, estado, funcao
  - [ ] palavras_chave (displayed as formatted list with | separators)
  - [ ] confianca (confidence percentage)
  - [ ] modelo_visao (which model analyzed this)
  - [ ] revisado_pelo_usuario (indicator if user reviewed)
  - [ ] Image preview (foto_original_path or recorte_path, if storage mounted)

### 5.3 Edit Object
- [ ] Open object details
- [ ] Click "Editar" / edit button
- [ ] Change a field (e.g., estado from "bom" to "danificado")
- [ ] Save changes
- [ ] Updated value reflects immediately in list and modal
- [ ] atualizado_em timestamp updates

### 5.4 Mark as Reviewed
- [ ] If revisado_pelo_usuario = 0, check for "Marcar como Revisado" button
- [ ] Click it
- [ ] revisado_pelo_usuario changes to 1
- [ ] UI indicator updates (e.g., checkmark or highlight)

### 5.5 Delete Object
- [ ] Open object details
- [ ] Click "Deletar" / delete button
- [ ] Confirmation dialog appears
- [ ] Object disappears from list after confirmation

---

## Section 6: Chat (AI Query)

### 6.1 Send Chat Message
- [ ] Click Chat section
- [ ] Type a question: "Quantas ferramentas há na cozinha?"
- [ ] Click "Enviar" / "Send" or press Enter
- [ ] Message appears in chat (user message on right, styled differently)

### 6.2 AI Response
- [ ] Wait for AI response (typically 2-5 seconds for Ollama, <1s for API)
- [ ] Response appears below user message (bot on left, styled differently)
- [ ] Response is contextual and makes sense (should reference actual data)
- [ ] Response is in Portuguese

### 6.3 Chat History Persistence
- [ ] Refresh page (F5 or Ctrl+R)
- [ ] Previous chat messages remain visible
- [ ] Full history loads from DB correctly

### 6.4 Mode-Specific Chat
- [ ] Switch operational mode (if available in config)
- [ ] Send same chat query
- [ ] Response should be similar but may vary in wording/depth
- [ ] Check logs for which model was used (modelo field in historico_chat table)

### 6.5 Long Chat Session
- [ ] Send 5-10 chat messages sequentially
- [ ] All messages and responses visible and correctly ordered
- [ ] No performance degradation as history grows
- [ ] Storage persists (check historico_chat table row count)

---

## Section 7: Estatísticas (Statistics)

### 7.1 Overall Statistics
- [ ] Navigate to Estatísticas section (if exists)
- [ ] Display shows:
  - [ ] Total objetos count
  - [ ] Objetos by category (bar chart or table)
  - [ ] Objects by status (bom/danificado/etc.)
  - [ ] Total localizações

### 7.2 Statistics After Processing
- [ ] Process new photo/video that creates objects
- [ ] Refresh statistics section
- [ ] New objects reflected in counts/charts
- [ ] Trends show correct increase

---

## Section 8: Observabilidade (Observability)

### 8.1 Health Endpoint (`/api/observabilidade/saude`)
- [ ] curl `http://localhost:8000/api/observabilidade/saude`
- [ ] Response includes:
  - [ ] status (string: "operacional" or similar)
  - [ ] modo (operational mode)
  - [ ] descricao (human-readable description)
  - [ ] timestamp (ISO 8601 format)

### 8.2 Metrics Endpoint (`/api/observabilidade/metricas`)
- [ ] curl `http://localhost:8000/api/observabilidade/metricas`
- [ ] Response includes:
  - [ ] fotos_processadas: counts by status (pendente, processando, concluído, erro)
  - [ ] videos_processados: counts by status
  - [ ] objetos_total (integer)
  - [ ] objetos_por_categoria (dict/array)
  - [ ] localizacoes_total (integer)

### 8.3 Logs Endpoint (`/api/observabilidade/logs`)
- [ ] curl `http://localhost:8000/api/observabilidade/logs?ultimos=10`
- [ ] Returns last 10 log entries
- [ ] curl with `?nivel=ERROR` returns only errors
- [ ] curl with `?logger=casaiq.migrations` returns only migration logs
- [ ] Timestamp, level, logger, message fields present

### 8.4 Summary Endpoint (`/api/observabilidade/resumo`)
- [ ] curl `http://localhost:8000/api/observabilidade/resumo`
- [ ] Response is aggregated: saude + metricas + logs_recentes (latest 20)

---

## Section 9: Database & Schema

### 9.1 Migrations Applied
- [ ] Start app with fresh DB (or delete casaiq.db)
- [ ] Check logs for migration messages: "Aplicando migração 001..."
- [ ] Verify `migrations` table has entry for version 1
- [ ] All schema tables created: localizacoes, categorias, objetos, fotos_processadas, videos_processados, historico_chat

### 9.2 Foreign Keys
- [ ] Create object with valid categoria_id and localizacao_id
- [ ] Try to delete a localizacao that has objects (should fail or cascade, depending on implementation)
- [ ] Verify referential integrity enforced in DB

### 9.3 Triggers
- [ ] Create an object
- [ ] Note the criado_em timestamp
- [ ] Wait 5 seconds
- [ ] Edit the object (e.g., change description)
- [ ] Verify atualizado_em is different (newer) than criado_em
- [ ] Check trigger `trg_objetos_atualizado_em` is working

---

## Section 10: Performance & Load

### 10.1 UI Responsiveness
- [ ] Upload a large photo (>5MB)
- [ ] UI remains responsive (buttons clickable, no freezing)
- [ ] Processing happens in background (can navigate while waiting)

### 10.2 Large Object Count
- [ ] If DB has 100+ objects, list still loads quickly (<2 seconds)
- [ ] Scroll through list smoothly

### 10.3 Search/Filter Performance
- [ ] Filter objects by nome with 50+ objects in list
- [ ] Results appear within 1 second

---

## Section 11: Error Recovery

### 11.1 Network Error
- [ ] Stop Ollama or API service mid-request
- [ ] Verify error message displays to user (not silent failure)
- [ ] User can retry after service resumes

### 11.2 DB Corruption
- [ ] Manually corrupt a record in DB (if safe)
- [ ] App should log error and continue (graceful degradation)
- [ ] Other users' requests not affected

### 11.3 Invalid Input
- [ ] Try to create location with empty nome field
- [ ] Form validation prevents submission (or backend rejects with 400 error)
- [ ] Error message displayed to user

---

## Section 12: Modo-Specific Testing

### 12.1 Ollama Local Mode
- [ ] Ensure `CASAIQ_MODO=ollama` (or auto-detect)
- [ ] Verify `/api/modo` shows "ollama"
- [ ] Chat and image processing use local model
- [ ] Latency is higher than API mode (expected)
- [ ] No external API calls in logs

### 12.2 Claude API Mode
- [ ] Set `ANTHROPIC_API_KEY` in environment
- [ ] Ensure `CASAIQ_MODO=anthropic` or auto-detected
- [ ] Chat and vision tasks use Claude API
- [ ] Responses are higher quality/faster than Ollama
- [ ] Check logs for API call signatures

### 12.3 OpenRouter Mode
- [ ] Set `OPENROUTER_API_KEY` in environment
- [ ] Ensure `CASAIQ_MODO=openrouter`
- [ ] Chat uses OpenRouter models
- [ ] Vision tasks may use different model selection logic
- [ ] Check /api/modo for correct vision_model and text_model

---

## Section 13: Browser Compatibility

- [ ] **Firefox:** All sections load, no console errors
- [ ] **Chrome/Chromium:** All sections load, no console errors
- [ ] **Safari (if macOS):** All sections load, no console errors
- [ ] **Mobile/Tablet:** Layout responds, touch interactions work

---

## Section 14: Logging & Debugging

### 14.1 Log Levels
- [ ] Check logs contain DEBUG, INFO, WARNING, ERROR entries
- [ ] Error log entries have full stack traces (if errors occur)
- [ ] JSON structured logging format is valid (parseable)

### 14.2 Observability During Processing
- [ ] Process a photo
- [ ] Check `/api/observabilidade/logs?nivel=INFO` mid-processing
- [ ] Logs show segmentation, analysis, enrichment stages

### 14.3 Database Log
- [ ] Check logs for database queries (if query logging enabled)
- [ ] No N+1 query patterns visible
- [ ] Indices being used efficiently

---

## Section 15: Final Sanity Checks

- [ ] No unhandled JavaScript exceptions in browser console
- [ ] No server 500 errors in logs
- [ ] All timestamps are correctly formatted (ISO 8601 or human-readable)
- [ ] All monetary/numeric values display correctly
- [ ] No hardcoded test data in production responses
- [ ] Graceful handling of missing static files (storage images not found, etc.)

---

## Test Result Template

When running full test suite, fill this template and keep as record:

```
Date: [YYYY-MM-DD]
Tester: [Name]
Mode: [ollama/anthropic/openrouter]
Duration: [minutes]

Results:
  Passed: [ ] / 150 checks
  Failed: [ ]
  Skipped: [ ] (explain why)

Critical Issues Found:
  (list any blocking issues)

Minor Issues Found:
  (list cosmetic/low-priority issues)

Notes:
  (any observations, performance notes, unusual behavior)

Signed: ________________   Date: __________
```

---

## Quick Reference: Common Test Scenarios

### Happy Path (5 min)
1. Create location → Upload photo → Check objects list → Chat query → View stats

### Error Path (5 min)
1. Upload invalid file → Try network error → Edit invalid object → Recovery

### Performance Path (10 min)
1. Load 100+ objects → Scroll list → Search → Filter → Chat with large dataset

### Mode Switching (15 min)
1. Test in Ollama mode → Switch to API mode → Compare response quality → Check logs

---

*Last tested: [DATE] by [TESTER]*
*Next review: [DATE]*
