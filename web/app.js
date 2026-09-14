/**
 * CasaIQ PRO — Frontend Controller v5.0
 * Interface industrial, veloz, resiliente e estável.
 */

const API = "/api";

// ─── Estado Global da Aplicação ─────────────────────────────────────────────
const state = {
  activeTab: "ingerir",
  selectedFile: null,
  selectedFileType: "foto", // 'foto' | 'video'
  locations: [],
  categories: [],
  inventoryItems: [],
  selectedItemIds: new Set(),
  currentStudioObjects: [],
  viewMode: "grid", // 'grid' | 'table'
  isRecordingVoice: false,
  recognition: null
};

// ─── Utilitário de Notificações Toast ───────────────────────────────────────
function showToast(message, type = "info", duration = 4000) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  const icon = type === "success" ? "✅" : type === "error" ? "❌" : "ℹ️";
  toast.innerHTML = `<span class="toast-icon">${icon}</span> <span>${message}</span>`;
  container.appendChild(toast);

  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transform = "translateX(100%)";
    toast.style.transition = "all 0.3s ease";
    setTimeout(() => toast.remove(), 300);
  }, duration);
}

// ─── HTTP Client Resiliente ────────────────────────────────────────────────
async function fetchJSON(endpoint, options = {}) {
  try {
    const res = await fetch(endpoint, options);
    if (!res.ok) {
      let errMsg = `HTTP ${res.status}`;
      try {
        const errObj = await res.json();
        errMsg = errObj.detail || errObj.message || errMsg;
      } catch {
        errMsg = await res.text();
      }
      throw new Error(errMsg);
    }
    return await res.json();
  } catch (err) {
    console.error(`Erro na requisição ${endpoint}:`, err);
    throw err;
  }
}

// ─── Inicialização da Aplicação ───────────────────────────────────────────
document.addEventListener("DOMContentLoaded", async () => {
  setupNavigation();
  setupDropzone();
  setupInventoryControls();
  setupChat();
  setupLocations();
  setupTelemetry();
  setupDrawer();

  // Garante que o drawer esteja fechado
  closeDrawer();

  // Carga inicial
  await Promise.allSettled([
    loadSystemMode(),
    loadLocations(),
    loadCategories(),
    loadInventory(),
    loadTelemetryMetrics()
  ]);
});

// ─── 1. Navegação por Abas ──────────────────────────────────────────────────
function switchTab(tabId) {
  document.querySelectorAll(".nav-tab").forEach(t => {
    t.classList.toggle("active", t.dataset.tab === tabId);
  });
  document.querySelectorAll(".tab-panel").forEach(p => {
    p.classList.toggle("active", p.id === `tab-${tabId}`);
  });
  state.activeTab = tabId;

  if (tabId === "inventario") loadInventory();
  if (tabId === "telemetria") loadTelemetryLogs();
  if (tabId === "localizacoes") loadLocations(true);
}

function setupNavigation() {
  document.querySelectorAll(".nav-tab").forEach(tab => {
    tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  });
}

// ─── 2. Status e Modo do Sistema ───────────────────────────────────────────
async function loadSystemMode() {
  try {
    const modo = await fetchJSON(`${API}/modo`);
    const txtModo = document.getElementById("txt-modo");
    if (txtModo) {
      const desc = modo.descricao || modo.modo || "Inteligente";
      const model = modo.vision_model ? ` (${modo.vision_model})` : "";
      txtModo.textContent = `${desc}${model}`;
    }
  } catch (e) {
    const txtModo = document.getElementById("txt-modo");
    if (txtModo) txtModo.textContent = "Modo: Local";
  }
}

// ─── 3. Ingestão (Dropzone, Upload & Studio) ─────────────────────────────────
function setupDropzone() {
  const dropzone = document.getElementById("dropzone");
  const fileInput = document.getElementById("input-foto");
  const emptyView = document.getElementById("dropzone-empty");
  const previewView = document.getElementById("dropzone-preview");
  const previewImg = document.getElementById("preview-img");
  const previewVid = document.getElementById("preview-vid");
  const previewName = document.getElementById("preview-filename");
  const btnRemover = document.getElementById("btn-remover-arquivo");
  const btnProcessar = document.getElementById("btn-processar");
  const form = document.getElementById("form-ingerir");
  const btnCamera = document.getElementById("btn-camera");
  const cameraInput = document.getElementById("input-camera");

  if (!dropzone || !fileInput) return;

  if (btnCamera && cameraInput) {
    btnCamera.addEventListener("click", () => cameraInput.click());
    cameraInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  dropzone.addEventListener("click", (e) => {
    if (e.target !== btnRemover && !state.selectedFile) {
      fileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => dropzone.classList.remove("dragover"));

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelected(e.dataTransfer.files[0]);
    }
  });

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelected(e.target.files[0]);
    }
  });

  btnRemover.addEventListener("click", (e) => {
    e.stopPropagation();
    state.selectedFile = null;
    fileInput.value = "";
    if (cameraInput) cameraInput.value = "";
    emptyView.hidden = false;
    previewView.hidden = true;
    previewImg.hidden = true;
    previewVid.hidden = true;
    btnProcessar.disabled = true;
  });

  function handleFileSelected(file) {
    state.selectedFile = file;
    const isVideo = file.type.startsWith("video/") || file.name.match(/\.(mp4|webm|mkv|mov)$/i);
    state.selectedFileType = isVideo ? "video" : "foto";

    previewName.textContent = `${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    emptyView.hidden = true;
    previewView.hidden = false;

    const url = URL.createObjectURL(file);
    if (isVideo) {
      previewImg.hidden = true;
      previewVid.hidden = false;
      previewVid.src = url;
    } else {
      previewVid.hidden = true;
      previewImg.hidden = false;
      previewImg.src = url;
    }
    btnProcessar.disabled = false;
  }

  // Submissão do Form de Ingestão
  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const locId = document.getElementById("select-localizacao").value;
    if (!locId || !state.selectedFile) {
      showToast("Selecione a localização e um arquivo.", "error");
      return;
    }

    btnProcessar.disabled = true;
    const progressBox = document.getElementById("progress-box");
    const progressFill = document.getElementById("progress-bar-fill");
    const progressTimer = document.getElementById("progress-timer");
    const progressDesc = document.getElementById("progress-step-desc");

    progressBox.hidden = false;
    progressFill.style.width = "10%";
    progressDesc.textContent = "Enviando arquivo para o servidor...";

    const startTime = performance.now();
    const timerInterval = setInterval(() => {
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(1);
      progressTimer.textContent = `${elapsed}s`;
    }, 100);

    const formData = new FormData();
    formData.append("localizacao_id", locId);
    formData.append("arquivo", state.selectedFile);

    const endpoint = state.selectedFileType === "video" ? `${API}/videos/ingerir` : `${API}/fotos/ingerir`;

    try {
      const res = await fetchJSON(endpoint, { method: "POST", body: formData });
      const taskDbId = state.selectedFileType === "video" ? res.video_id : res.foto_id;
      showToast("Arquivo enviado! Processando pipeline de 4 agentes...", "info");

      // Polling de Status
      pollProcessingStatus(taskDbId, state.selectedFileType, () => {
        clearInterval(timerInterval);
        progressFill.style.width = "100%";
        progressDesc.textContent = "Processamento concluído com sucesso!";
        showToast("Pipeline finalizado com sucesso!", "success");
        btnProcessar.disabled = false;
        loadStudioResults(taskDbId);
        loadInventory();
        loadTelemetryMetrics();
      }, (err) => {
        clearInterval(timerInterval);
        progressDesc.textContent = `Erro: ${err.message}`;
        showToast(`Falha no processamento: ${err.message}`, "error");
        btnProcessar.disabled = false;
      });

    } catch (err) {
      clearInterval(timerInterval);
      progressBox.hidden = true;
      btnProcessar.disabled = false;
      showToast(`Erro no upload: ${err.message}`, "error");
    }
  });

  // Modal Rápido de Nova Localização
  const btnQuickLoc = document.getElementById("btn-quick-new-loc");
  const modalQuickLoc = document.getElementById("modal-quick-loc");
  const formQuickLoc = document.getElementById("form-quick-loc");
  const btnCloseQuickLoc = document.getElementById("btn-close-quick-loc");

  if (btnQuickLoc && modalQuickLoc) {
    btnQuickLoc.addEventListener("click", () => modalQuickLoc.showModal());
    btnCloseQuickLoc.addEventListener("click", () => modalQuickLoc.close());
    formQuickLoc.addEventListener("submit", async (e) => {
      e.preventDefault();
      const nome = document.getElementById("quick-loc-nome").value.trim();
      const comodo = document.getElementById("quick-loc-comodo").value.trim();
      try {
        const nova = await fetchJSON(`${API}/localizacoes`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ nome, comodo, tipo: "caixa" })
        });
        showToast(`Localização "${nova.nome}" criada!`, "success");
        await loadLocations();
        document.getElementById("select-localizacao").value = nova.id;
        formQuickLoc.reset();
        modalQuickLoc.close();
      } catch (err) {
        showToast(`Erro ao criar: ${err.message}`, "error");
      }
    });
  }
}

// Polling de Status do Pipeline
function pollProcessingStatus(id, type, onComplete, onError) {
  const statusEndpoint = type === "video" ? `${API}/videos/${id}/status` : `${API}/fotos/${id}/status`;
  const fill = document.getElementById("progress-bar-fill");
  const desc = document.getElementById("progress-step-desc");

  const interval = setInterval(async () => {
    try {
      const s = await fetchJSON(statusEndpoint);
      if (s.status === "concluido") {
        clearInterval(interval);
        onComplete(s);
      } else if (s.status === "erro") {
        clearInterval(interval);
        onError(new Error(s.erro_mensagem || "Erro no pipeline"));
      } else {
        let textoProgresso = "Processando pipeline...";
        if (s.progresso) {
          try {
            const pj = typeof s.progresso === "string" ? JSON.parse(s.progresso) : s.progresso;
            textoProgresso = pj.descricao || pj.etapa || s.progresso;
          } catch {
            textoProgresso = s.progresso;
          }
        }
        desc.textContent = textoProgresso;
        const currentPct = parseInt(fill.style.width) || 10;
        if (currentPct < 90) fill.style.width = `${currentPct + 15}%`;
      }
    } catch (e) {
      clearInterval(interval);
      onError(e);
    }
  }, 1200);
}

// Carga do Studio de Inspeção Visual (Canvas + Overlays SVG)
async function loadStudioResults(fotoId) {
  try {
    const foto = await fetchJSON(`${API}/fotos/${fotoId}/status`);
    const objetos = await fetchJSON(`${API}/fotos/${fotoId}/objetos`);
    state.currentStudioObjects = objetos;

    const canvasWrapper = document.getElementById("canvas-wrapper");
    const baseImg = document.getElementById("studio-base-img");
    const countBadge = document.getElementById("detection-count");
    const container = document.getElementById("detected-objects-container");

    countBadge.textContent = `${objetos.length} detectados`;
    canvasWrapper.hidden = false;

    if (foto.caminho) {
      baseImg.src = `/${foto.caminho}`;
      baseImg.onload = () => renderSvgOverlays(baseImg, objetos);
    }

    container.innerHTML = "";
    if (!objetos.length) {
      container.innerHTML = `<div class="empty-state-card"><p>Nenhum objeto detectado nesta imagem.</p></div>`;
      return;
    }

    objetos.forEach(obj => {
      const card = createObjectCard(obj, false);
      container.appendChild(card);
    });
  } catch (err) {
    console.error("Erro ao carregar studio:", err);
  }
}

function renderSvgOverlays(imgElem, objetos) {
  const svg = document.getElementById("studio-svg-overlay");
  svg.innerHTML = "";
  const rect = imgElem.getBoundingClientRect();
  const W = imgElem.naturalWidth || rect.width || 800;
  const H = imgElem.naturalHeight || rect.height || 600;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  objetos.forEach((obj) => {
    // 1. Polígono GeoJSON se disponível
    if (obj.geometria_vetor) {
      try {
        const geo = typeof obj.geometria_vetor === "string" ? JSON.parse(obj.geometria_vetor) : obj.geometria_vetor;
        if (geo && geo.coordinates) {
          const coords = geo.coordinates[0] || geo.coordinates;
          const points = coords.map(pt => `${pt[0]},${pt[1]}`).join(" ");
          const polygon = document.createElementNS("http://www.w3.org/2000/svg", "polygon");
          polygon.setAttribute("points", points);
          polygon.setAttribute("class", "svg-polygon");
          polygon.dataset.objId = obj.id;
          svg.appendChild(polygon);
        }
      } catch (e) {}
    }

    // 2. Bounding Box
    if (obj.bbox) {
      const b = obj.bbox; // {ymin, xmin, ymax, xmax} normalizado [0,1]
      const rx = b.xmin * W;
      const ry = b.ymin * H;
      const rw = (b.xmax - b.xmin) * W;
      const rh = (b.ymax - b.ymin) * H;

      const rectNode = document.createElementNS("http://www.w3.org/2000/svg", "rect");
      rectNode.setAttribute("x", rx);
      rectNode.setAttribute("y", ry);
      rectNode.setAttribute("width", rw);
      rectNode.setAttribute("height", rh);
      rectNode.setAttribute("class", "svg-bbox");
      rectNode.dataset.objId = obj.id;
      rectNode.addEventListener("click", () => openDrawer(obj.id));
      svg.appendChild(rectNode);
    }
  });
}

// ─── 4. Inventário (Filtros, Grid, Tabela & Ações em Lote) ───────────────────
function setupInventoryControls() {
  const searchInput = document.getElementById("inv-search");
  const clearBtn = document.getElementById("btn-clear-search");
  const filterCat = document.getElementById("inv-filtro-categoria");
  const filterLoc = document.getElementById("inv-filtro-localizacao");
  const filterEst = document.getElementById("inv-filtro-estado");
  const sortSelect = document.getElementById("inv-sort");
  const btnRefresh = document.getElementById("btn-refresh-inv");
  const viewToggleBtns = document.querySelectorAll(".view-toggle-btn");
  const chkSelectAll = document.getElementById("chk-select-all");

  let debounceTimer;
  searchInput.addEventListener("input", () => {
    clearBtn.hidden = !searchInput.value;
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => loadInventory(), 250);
  });

  clearBtn.addEventListener("click", () => {
    searchInput.value = "";
    clearBtn.hidden = true;
    loadInventory();
  });

  [filterCat, filterLoc, filterEst, sortSelect].forEach(el => {
    if (el) el.addEventListener("change", () => loadInventory());
  });

  if (btnRefresh) btnRefresh.addEventListener("click", () => loadInventory());

  viewToggleBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      viewToggleBtns.forEach(b => b.classList.remove("active"));
      btn.classList.add("active");
      state.viewMode = btn.dataset.mode;
      renderInventory();
    });
  });

  // Ações em Lote
  const btnBulkMove = document.getElementById("btn-bulk-move");
  const btnBulkDelete = document.getElementById("btn-bulk-delete");
  const btnBulkCancel = document.getElementById("btn-bulk-cancel");

  if (btnBulkCancel) {
    btnBulkCancel.addEventListener("click", () => {
      state.selectedItemIds.clear();
      updateBulkBar();
      renderInventory();
    });
  }

  if (chkSelectAll) {
    chkSelectAll.addEventListener("change", (e) => {
      if (e.target.checked) {
        state.inventoryItems.forEach(i => state.selectedItemIds.add(i.id));
      } else {
        state.selectedItemIds.clear();
      }
      updateBulkBar();
      renderInventory();
    });
  }

  if (btnBulkDelete) {
    btnBulkDelete.addEventListener("click", async () => {
      const ids = Array.from(state.selectedItemIds);
      if (!ids.length || !confirm(`Excluir permanentemente ${ids.length} objetos selecionados?`)) return;
      try {
        await fetchJSON(`${API}/objetos/batch-delete`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids })
        });
        showToast(`${ids.length} objetos excluídos!`, "success");
        state.selectedItemIds.clear();
        updateBulkBar();
        loadInventory();
        loadTelemetryMetrics();
      } catch (e) {
        showToast(`Erro ao excluir: ${e.message}`, "error");
      }
    });
  }

  if (btnBulkMove) {
    btnBulkMove.addEventListener("click", async () => {
      const ids = Array.from(state.selectedItemIds);
      const targetLoc = document.getElementById("bulk-move-target").value;
      if (!ids.length || !targetLoc) {
        showToast("Selecione a localização de destino.", "error");
        return;
      }
      try {
        await fetchJSON(`${API}/objetos/batch-move`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids, localizacao_id: parseInt(targetLoc) })
        });
        showToast(`${ids.length} objetos movidos!`, "success");
        state.selectedItemIds.clear();
        updateBulkBar();
        loadInventory();
      } catch (e) {
        showToast(`Erro ao mover: ${e.message}`, "error");
      }
    });
  }
}

function updateBulkBar() {
  const bulkBar = document.getElementById("bulk-bar");
  const countSpan = document.getElementById("bulk-selected-count");
  const totalSelected = state.selectedItemIds.size;
  if (countSpan) countSpan.textContent = totalSelected;
  if (bulkBar) bulkBar.hidden = totalSelected === 0;
}

async function loadInventory() {
  const params = new URLSearchParams();
  const search = document.getElementById("inv-search")?.value.trim() || "";
  const cat = document.getElementById("inv-filtro-categoria")?.value || "";
  const loc = document.getElementById("inv-filtro-localizacao")?.value || "";
  const est = document.getElementById("inv-filtro-estado")?.value || "";

  if (search) params.set("busca", search);
  if (cat) params.set("categoria", cat);
  if (loc) params.set("localizacao", loc);
  if (est) params.set("estado", est);

  try {
    const items = await fetchJSON(`${API}/objetos?${params.toString()}`);
    const sort = document.getElementById("inv-sort")?.value || "recent";
    if (sort === "name_asc") items.sort((a, b) => a.nome.localeCompare(b.nome));
    else if (sort === "conf_desc") items.sort((a, b) => (b.confianca || 0) - (a.confianca || 0));
    else if (sort === "weight_desc") items.sort((a, b) => (b.peso_estimado_g || 0) - (a.peso_estimado_g || 0));

    state.inventoryItems = items;
    renderInventory();
  } catch (err) {
    console.error("Erro ao carregar inventário:", err);
  }
}

function renderInventory() {
  const gridView = document.getElementById("inv-grid-view");
  const tableView = document.getElementById("inv-table-view");
  const tableBody = document.getElementById("inv-table-body");

  if (!gridView || !tableView || !tableBody) return;

  if (!state.inventoryItems.length) {
    const emptyHTML = `
      <div class="empty-state-card">
        <div class="empty-icon-glow">📦</div>
        <h3 class="empty-title">Seu inventário está vazio</h3>
        <p class="empty-desc">Nenhum objeto foi cadastrado ainda. Comece ingerindo uma foto ou vídeo para que a inteligência artificial faça a segmentação e catalogação automática.</p>
        <button type="button" class="btn-action-primary" id="btn-empty-go-ingerir">
          <span>📸 Fazer Primeira Ingestão</span>
        </button>
      </div>
    `;
    gridView.hidden = false;
    tableView.hidden = true;
    gridView.innerHTML = emptyHTML;

    document.getElementById("btn-empty-go-ingerir")?.addEventListener("click", () => switchTab("ingerir"));
    return;
  }

  if (state.viewMode === "grid") {
    gridView.hidden = false;
    tableView.hidden = true;
    gridView.innerHTML = "";

    state.inventoryItems.forEach(item => {
      const card = createObjectCard(item, true);
      gridView.appendChild(card);
    });
  } else {
    gridView.hidden = true;
    tableView.hidden = false;
    tableBody.innerHTML = "";

    state.inventoryItems.forEach(item => {
      const tr = document.createElement("tr");
      const isSelected = state.selectedItemIds.has(item.id);
      const imgUrl = item.icone_path ? `/storage/icones/${item.icone_path.split("/").pop()}` : "";

      tr.innerHTML = `
        <td><input type="checkbox" class="chk-item-select" data-id="${item.id}" ${isSelected ? "checked" : ""}></td>
        <td>${imgUrl ? `<img src="${imgUrl}" class="tbl-thumb" alt="${item.nome}" />` : "📦"}</td>
        <td><strong>${item.nome}</strong></td>
        <td>${item.categoria_icone || "📦"} ${item.categoria_nome || "—"}</td>
        <td>📍 ${item.localizacao_nome || "—"}</td>
        <td>${item.tamanho_estimado_cm || "—"} / ${item.peso_estimado_g ? item.peso_estimado_g + "g" : "—"}</td>
        <td>${item.material || "—"}</td>
        <td>${renderColorDots(item.cores_json)}</td>
        <td><span class="conf-rate">${((item.confianca || 1.0) * 100).toFixed(0)}%</span></td>
        <td>
          <button type="button" class="btn-details-card btn-open-drawer" data-id="${item.id}">Ver / Editar</button>
        </td>
      `;

      tr.querySelector(".chk-item-select").addEventListener("change", (e) => {
        if (e.target.checked) state.selectedItemIds.add(item.id);
        else state.selectedItemIds.delete(item.id);
        updateBulkBar();
      });

      tr.querySelector(".btn-open-drawer").addEventListener("click", () => openDrawer(item.id));
      tableBody.appendChild(tr);
    });
  }
}

function createObjectCard(obj, allowSelection = true) {
  const card = document.createElement("div");
  card.className = "obj-card";
  const isSelected = state.selectedItemIds.has(obj.id);
  const imgUrl = obj.icone_path ? `/storage/icones/${obj.icone_path.split("/").pop()}` : "";

  card.innerHTML = `
    <div class="card-img-wrap">
      ${allowSelection ? `
        <div class="card-checkbox-wrap">
          <input type="checkbox" class="chk-item-select" data-id="${obj.id}" ${isSelected ? "checked" : ""}>
        </div>
      ` : ""}
      <span class="card-cat-badge">${obj.categoria_nome || "Geral"}</span>
      ${imgUrl ? `<img src="${imgUrl}" class="card-real-img" alt="${obj.nome}" />` : `<div style="font-size:2.8rem; opacity:0.6;">📦</div>`}
    </div>
    <div class="card-body-wrap">
      <div class="card-obj-title" title="${obj.nome}">${obj.nome}</div>
      <div class="card-obj-loc">📍 ${obj.localizacao_nome || "Sem Local"} ${obj.localizacao_comodo ? `(${obj.localizacao_comodo})` : ""}</div>
      <div class="card-palette-row">${renderColorDots(obj.cores_json)}</div>
      <div class="card-bottom-footer">
        <span class="conf-rate">Conf: ${((obj.confianca || 1.0) * 100).toFixed(0)}%</span>
        <button type="button" class="btn-details-card btn-open-drawer" data-id="${obj.id}">Detalhes ↗</button>
      </div>
    </div>
  `;

  if (allowSelection) {
    card.querySelector(".chk-item-select").addEventListener("change", (e) => {
      if (e.target.checked) state.selectedItemIds.add(obj.id);
      else state.selectedItemIds.delete(obj.id);
      updateBulkBar();
    });
  }

  card.querySelector(".btn-open-drawer").addEventListener("click", () => openDrawer(obj.id));
  return card;
}

function renderColorDots(coresJson) {
  if (!coresJson) return "";
  try {
    const list = typeof coresJson === "string" ? JSON.parse(coresJson) : coresJson;
    if (!Array.isArray(list)) return "";
    return list.slice(0, 4).map(c => `<span class="color-dot-circle" style="background-color: ${c.hex || '#666'};" title="${c.nome || c.hex}"></span>`).join("");
  } catch {
    return "";
  }
}

// ─── 5. Drawer de Detalhes do Objeto ────────────────────────────────────────
function setupDrawer() {
  const backdrop = document.getElementById("drawer-backdrop");
  const btnClose = document.getElementById("btn-close-drawer");
  const form = document.getElementById("form-drawer-edit");
  const btnDelete = document.getElementById("btn-drawer-delete");

  if (btnClose) btnClose.addEventListener("click", closeDrawer);
  if (backdrop) backdrop.addEventListener("click", closeDrawer);

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const id = document.getElementById("drawer-obj-id").value;
      const payload = {
        nome: document.getElementById("drawer-edit-nome").value.trim(),
        categoria_id: parseInt(document.getElementById("drawer-edit-categoria").value) || null,
        localizacao_id: parseInt(document.getElementById("drawer-edit-localizacao").value) || null,
        tamanho: document.getElementById("drawer-edit-tamanho").value.trim() || null,
        peso_estimado_g: parseInt(document.getElementById("drawer-edit-peso").value) || null,
        material: document.getElementById("drawer-edit-material").value.trim() || null,
        estado: document.getElementById("drawer-edit-estado").value,
        funcao: document.getElementById("drawer-edit-funcao").value.trim() || null
      };

      try {
        await fetchJSON(`${API}/objetos/${id}`, {
          method: "PUT",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });
        showToast("Objeto atualizado com sucesso!", "success");
        closeDrawer();
        loadInventory();
      } catch (err) {
        showToast(`Erro ao salvar: ${err.message}`, "error");
      }
    });
  }

  if (btnDelete) {
    btnDelete.addEventListener("click", async () => {
      const id = document.getElementById("drawer-obj-id").value;
      if (!confirm("Excluir permanentemente este objeto?")) return;
      try {
        await fetchJSON(`${API}/objetos/${id}`, { method: "DELETE" });
        showToast("Objeto excluído!", "success");
        closeDrawer();
        loadInventory();
        loadTelemetryMetrics();
      } catch (err) {
        showToast(`Erro ao excluir: ${err.message}`, "error");
      }
    });
  }
}

async function openDrawer(objId) {
  try {
    const obj = await fetchJSON(`${API}/objetos/${objId}`);
    document.getElementById("drawer-obj-id").value = obj.id;
    document.getElementById("drawer-obj-name").textContent = obj.nome;
    document.getElementById("drawer-edit-nome").value = obj.nome || "";
    document.getElementById("drawer-edit-tamanho").value = obj.tamanho_estimado_cm || obj.tamanho || "";
    document.getElementById("drawer-edit-peso").value = obj.peso_estimado_g || "";
    document.getElementById("drawer-edit-material").value = obj.material || "";
    document.getElementById("drawer-edit-estado").value = obj.estado || "bom";
    document.getElementById("drawer-edit-funcao").value = obj.funcao || "";

    const iconeImg = document.getElementById("drawer-icone-img");
    const iconePh = document.getElementById("drawer-icone-placeholder");
    if (obj.icone_path) {
      iconeImg.src = `/storage/icones/${obj.icone_path.split("/").pop()}`;
      iconeImg.hidden = false;
      iconePh.hidden = true;
    } else {
      iconeImg.hidden = true;
      iconePh.hidden = false;
    }

    const recorteImg = document.getElementById("drawer-recorte-img");
    const recortePh = document.getElementById("drawer-recorte-placeholder");
    if (obj.recorte_path) {
      recorteImg.src = `/${obj.recorte_path}`;
      recorteImg.hidden = false;
      recortePh.hidden = true;
    } else {
      recorteImg.hidden = true;
      recortePh.hidden = false;
    }

    document.getElementById("drawer-icone-fonte").textContent = obj.icone_fonte || "Ícone";

    const selCat = document.getElementById("drawer-edit-categoria");
    const selLoc = document.getElementById("drawer-edit-localizacao");
    populateSelect(selCat, state.categories, obj.categoria_id, "id", "nome");
    populateSelect(selLoc, state.locations, obj.localizacao_id, "id", "nome");

    const colorsDiv = document.getElementById("drawer-colors-palette");
    colorsDiv.innerHTML = "";
    if (obj.cores_json) {
      try {
        const cores = JSON.parse(obj.cores_json);
        cores.forEach(c => {
          colorsDiv.innerHTML += `
            <div class="color-chip">
              <span class="color-dot-circle" style="background-color: ${c.hex}"></span>
              <span>${c.nome || c.hex} (${c.area_pct || 0}%)</span>
            </div>
          `;
        });
      } catch (e) {}
    }

    const polyDiv = document.getElementById("drawer-polygon-preview");
    polyDiv.innerHTML = obj.geometria_vetor ? `<span style="font-size:0.75rem; color:var(--emerald-success);">✅ Polígono GeoJSON Vetorial Presente</span>` : `<span style="font-size:0.75rem; color:var(--text-dim);">Nenhum polígono extraído</span>`;

    document.getElementById("drawer-backdrop").hidden = false;
    document.getElementById("object-drawer").hidden = false;
  } catch (err) {
    showToast(`Erro ao abrir objeto: ${err.message}`, "error");
  }
}

function closeDrawer() {
  const backdrop = document.getElementById("drawer-backdrop");
  const drawer = document.getElementById("object-drawer");
  if (backdrop) backdrop.hidden = true;
  if (drawer) drawer.hidden = true;
}

// ─── 6. Assistente & Chat Pro com Voz ───────────────────────────────────────
function setupChat() {
  const form = document.getElementById("form-chat");
  const input = document.getElementById("chat-input-text");
  const messagesBox = document.getElementById("chat-messages");
  const voiceBtn = document.getElementById("btn-voice-input");

  document.querySelectorAll(".prompt-pill").forEach(pill => {
    pill.addEventListener("click", () => {
      input.value = pill.dataset.prompt;
      form.dispatchEvent(new Event("submit"));
    });
  });

  if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    state.recognition = new SpeechRecognition();
    state.recognition.lang = "pt-BR";
    state.recognition.continuous = false;

    state.recognition.onstart = () => {
      state.isRecordingVoice = true;
      voiceBtn.classList.add("recording");
      showToast("Ouvindo... Fale sua pergunta.", "info", 2000);
    };

    state.recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      input.value = transcript;
      form.dispatchEvent(new Event("submit"));
    };

    state.recognition.onend = () => {
      state.isRecordingVoice = false;
      voiceBtn.classList.remove("recording");
    };

    voiceBtn.addEventListener("click", () => {
      if (state.isRecordingVoice) state.recognition.stop();
      else state.recognition.start();
    });
  } else {
    if (voiceBtn) voiceBtn.style.display = "none";
  }

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const pergunta = input.value.trim();
      if (!pergunta) return;

      appendChatBubble("user", pergunta);
      input.value = "";

      const loadingBubble = appendChatBubble("assistant", "Pensando e consultando base de dados...");

      try {
        const res = await fetchJSON(`${API}/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ pergunta })
        });

        loadingBubble.remove();
        appendChatBubble("assistant", res.resposta, res.sql_executado, res.modelo);

        if ("speechSynthesis" in window) {
          const utter = new SpeechSynthesisUtterance(res.resposta.replace(/[*#_]/g, ""));
          utter.lang = "pt-BR";
          window.speechSynthesis.speak(utter);
        }
      } catch (err) {
        loadingBubble.remove();
        appendChatBubble("assistant", `⚠️ Desculpe, ocorreu um erro: ${err.message}`);
      }
    });
  }

  function appendChatBubble(role, text, sql = null, model = null) {
    const bubble = document.createElement("div");
    bubble.className = `chat-msg ${role}`;
    const avatar = role === "user" ? "👤" : "👑";
    const author = role === "user" ? "Você" : `Assistente CasaIQ (${model || "Local"})`;

    bubble.innerHTML = `
      <div class="msg-avatar">${avatar}</div>
      <div class="msg-content">
        <div class="msg-author">${author}</div>
        <div class="msg-text">${text}</div>
        ${sql ? `
          <div class="sql-view-box">
            <details>
              <summary style="cursor:pointer; color:var(--text-dim);">🔍 Ver Consulta SQL Executada</summary>
              <div class="sql-snippet">${sql}</div>
            </details>
          </div>
        ` : ""}
      </div>
    `;

    messagesBox.appendChild(bubble);
    messagesBox.scrollTop = messagesBox.scrollHeight;
    return bubble;
  }
}

// ─── 7. Gestão de Localizações & Reset ──────────────────────────────────────
function setupLocations() {
  const form = document.getElementById("form-localizacao");
  const btnReset = document.getElementById("btn-danger-reset");

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const nome = document.getElementById("loc-nome").value.trim();
      const tipo = document.getElementById("loc-tipo").value;
      const comodo = document.getElementById("loc-comodo").value.trim();
      const descricao = document.getElementById("loc-descricao").value.trim();

      try {
        await fetchJSON(`${API}/localizacoes`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ nome, tipo, comodo, descricao })
        });
        showToast(`Localização "${nome}" adicionada com sucesso!`, "success");
        form.reset();
        loadLocations(true);
      } catch (err) {
        showToast(`Erro: ${err.message}`, "error");
      }
    });
  }

  if (btnReset) {
    btnReset.addEventListener("click", async () => {
      const confirmText = prompt("⚠️ AVISO: Digite exatamente 'ZERAR' para apagar todos os objetos e dados:");
      if (confirmText !== "ZERAR") {
        showToast("Operação cancelada.", "info");
        return;
      }
      try {
        await fetchJSON(`${API}/admin/resetar-banco`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ confirmacao: "ZERAR" })
        });
        showToast("Todos os dados foram zerados com sucesso!", "success");
        await Promise.all([
          loadLocations(true),
          loadInventory(),
          loadTelemetryMetrics()
        ]);
      } catch (err) {
        showToast(`Erro ao resetar: ${err.message}`, "error");
      }
    });
  }
}

async function loadLocations(detailed = false) {
  try {
    const locs = await fetchJSON(`${API}/localizacoes`);
    state.locations = locs;

    const selIngerir = document.getElementById("select-localizacao");
    const selFiltro = document.getElementById("inv-filtro-localizacao");
    const selBulk = document.getElementById("bulk-move-target");

    populateSelect(selIngerir, locs, locs[0]?.id || null, "id", "nome", "Selecione uma localização...");
    populateSelect(selFiltro, locs, null, "id", "nome", "Todas as Localizações");
    populateSelect(selBulk, locs, null, "id", "nome", "Mover para...");

    if (detailed) {
      const grid = document.getElementById("localizacoes-grid");
      const count = document.getElementById("loc-total-count");
      if (count) count.textContent = `${locs.length} locais`;
      if (grid) {
        grid.innerHTML = "";
        if (!locs.length) {
          grid.innerHTML = `<div class="empty-state-card"><p>Nenhum local cadastrado.</p></div>`;
          return;
        }
        locs.forEach(l => {
          const card = document.createElement("div");
          card.className = "card-box";
          card.style.padding = "1rem";
          card.innerHTML = `
            <div style="display:flex; justify-content:space-between; align-items:center;">
              <div>
                <strong style="font-size:0.95rem;">📍 ${l.nome}</strong>
                <div style="font-size:0.75rem; color:var(--text-dim);">${l.comodo || "Sem Cômodo"} · Tipo: ${l.tipo}</div>
              </div>
              <span class="badge-count">${l.total_objetos || 0} objetos</span>
            </div>
            ${l.descricao ? `<p style="font-size:0.78rem; color:var(--text-muted); margin-top:0.4rem;">${l.descricao}</p>` : ""}
            <div style="margin-top:0.75rem; text-align:right;">
              <button class="btn-danger-outline" style="padding:0.25rem 0.6rem; font-size:0.75rem;" data-id="${l.id}">Excluir</button>
            </div>
          `;

          card.querySelector("button").addEventListener("click", async () => {
            if (!confirm(`Excluir localização "${l.nome}"?`)) return;
            try {
              await fetchJSON(`${API}/localizacoes/${l.id}`, { method: "DELETE" });
              showToast("Localização excluída!", "success");
              loadLocations(true);
            } catch (e) {
              showToast(`Erro: ${e.message}`, "error");
            }
          });

          grid.appendChild(card);
        });
      }
    }
  } catch (err) {
    console.error("Erro ao carregar localizações:", err);
  }
}

async function loadCategories() {
  try {
    const cats = await fetchJSON(`${API}/categorias`);
    state.categories = cats;
    const selFiltro = document.getElementById("inv-filtro-categoria");
    populateSelect(selFiltro, cats, null, "id", "nome", "Todas as Categorias", "icone");
  } catch (e) {
    console.warn("Rota /api/categorias ausente ou erro.");
  }
}

function populateSelect(selectElem, items, selectedId, valKey, labelKey, placeholder = null, iconKey = null) {
  if (!selectElem) return;
  selectElem.innerHTML = "";
  if (placeholder) {
    selectElem.innerHTML += `<option value="">${placeholder}</option>`;
  }
  items.forEach(it => {
    const isSel = selectedId !== null && it[valKey] === selectedId ? "selected" : "";
    const prefix = iconKey && it[iconKey] ? `${it[iconKey]} ` : "";
    selectElem.innerHTML += `<option value="${it[valKey]}" ${isSel}>${prefix}${it[labelKey]}</option>`;
  });
}

// ─── 8. Telemetria & Observabilidade ────────────────────────────────────────
function setupTelemetry() {
  const btnRefresh = document.getElementById("btn-refresh-logs");
  if (btnRefresh) btnRefresh.addEventListener("click", loadTelemetryLogs);
}

async function loadTelemetryMetrics() {
  try {
    const stats = await fetchJSON(`${API}/estatisticas`);
    const totalObjs = stats.total_objetos || 0;
    document.getElementById("m-total-objetos").textContent = totalObjs;
    document.getElementById("m-total-fotos").textContent = stats.total_fotos || 0;
    document.getElementById("txt-db").textContent = `${totalObjs} objetos`;
  } catch (e) {}
}

async function loadTelemetryLogs() {
  const terminal = document.getElementById("logs-terminal");
  if (!terminal) return;
  try {
    const logs = await fetchJSON(`${API}/observabilidade/logs`);
    terminal.innerHTML = "";
    if (!logs.length) {
      terminal.innerHTML = `<div class="log-row info">[SISTEMA] Nenhum log recente capturado.</div>`;
      return;
    }
    logs.slice(-30).forEach(l => {
      const cls = l.level === "ERROR" ? "error" : l.level === "WARNING" ? "warn" : "info";
      terminal.innerHTML += `
        <div class="log-row ${cls}">
          <span style="color:var(--text-dim);">${l.timestamp.split("T")[1].slice(0, 8)}</span>
          <strong>[${l.logger}]</strong> ${l.message}
        </div>
      `;
    });
  } catch (e) {
    terminal.innerHTML = `<div class="log-row warn">[AVISO] Rota de logs indisponível.</div>`;
  }
}
