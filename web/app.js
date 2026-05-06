// CasaIQ — interface (vanilla JS)
const API = "/api";

// ─── Navegação entre abas ────────────────────────────────────────────────────
document.querySelectorAll(".aba").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".aba").forEach(b => b.classList.remove("ativa"));
    document.querySelectorAll(".painel").forEach(p => p.classList.remove("ativo"));
    btn.classList.add("ativa");
    document.getElementById("aba-" + btn.dataset.aba).classList.add("ativo");
    if (btn.dataset.aba === "inventario") carregarInventario();
    if (btn.dataset.aba === "localizacoes") carregarLocalizacoes(true);
  });
});

// ─── Util ────────────────────────────────────────────────────────────────────
async function fetchJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`HTTP ${r.status}: ${await r.text()}`);
  return r.json();
}

function badgeFonte(fonte) {
  const mapa = {
    recorte: "📷 recorte", web: "🌐 web",
    claude_desenho: "✏️ claude", placeholder: "📦 placeholder",
    recorte_baixa_qualidade: "📷 recorte (baixa)",
  };
  return mapa[fonte] || fonte || "—";
}

function badgeModelo(modelo) {
  if (modelo === "claude_api") return '<span class="badge claude">☁️ Claude API</span>';
  return '<span class="badge ollama">💻 Ollama</span>';
}

// ─── Modo de operação (rodapé) ──────────────────────────────────────────────
async function carregarModo() {
  try {
    const m = await fetchJSON(`${API}/modo`);
    const partes = [`Modo: ${m.descricao}`];
    if (m.vision_model) partes.push(`👁️ ${m.vision_model}`);
    if (m.text_model)   partes.push(`💬 ${m.text_model}`);
    if (m.timeout_s)    partes.push(`⏱️ ${m.timeout_s}s`);
    document.getElementById("badge-modo").textContent = partes.join(" · ");
  } catch (e) {
    document.getElementById("badge-modo").textContent = "Modo: indisponível";
  }
}

// ─── Localizações ────────────────────────────────────────────────────────────
async function carregarLocalizacoes(detalhada = false) {
  const locs = await fetchJSON(`${API}/localizacoes`);
  const sel = document.getElementById("select-localizacao");
  sel.innerHTML = '<option value="">Selecione...</option>';
  for (const l of locs) {
    sel.innerHTML += `<option value="${l.id}">${l.nome}${l.comodo ? ` — ${l.comodo}` : ""}</option>`;
  }
  // filtro do inventário
  const filtroLoc = document.getElementById("filtro-localizacao");
  filtroLoc.innerHTML = '<option value="">Todas as localizações</option>';
  for (const l of locs) {
    filtroLoc.innerHTML += `<option value="${l.id}">${l.nome}</option>`;
  }
  if (detalhada) {
    const ul = document.getElementById("lista-localizacoes");
    ul.innerHTML = "";
    for (const l of locs) {
      const li = document.createElement("li");
      li.innerHTML = `
        <span class="info">
          <span class="nome">${l.nome}</span>
          <span class="meta">${l.tipo}${l.comodo ? " · " + l.comodo : ""} · ${l.total_objetos} obj.</span>
        </span>
        <button data-id="${l.id}">Excluir</button>
      `;
      li.querySelector("button").addEventListener("click", async () => {
        if (!confirm(`Excluir "${l.nome}"?`)) return;
        try {
          await fetchJSON(`${API}/localizacoes/${l.id}`, { method: "DELETE" });
          carregarLocalizacoes(true);
        } catch (e) {
          alert("Erro: " + e.message);
        }
      });
      ul.appendChild(li);
    }
  }
}

document.getElementById("form-localizacao").addEventListener("submit", async e => {
  e.preventDefault();
  const dados = {
    nome:      document.getElementById("loc-nome").value.trim(),
    tipo:      document.getElementById("loc-tipo").value.trim() || "caixa",
    comodo:    document.getElementById("loc-comodo").value.trim() || null,
    descricao: document.getElementById("loc-descricao").value.trim() || null,
  };
  try {
    await fetchJSON(`${API}/localizacoes`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(dados),
    });
    e.target.reset();
    document.getElementById("loc-tipo").value = "caixa";
    carregarLocalizacoes(true);
  } catch (err) {
    alert("Erro: " + err.message);
  }
});

// ─── Ingestão de fotos e vídeos ──────────────────────────────────────────────
const dropzone      = document.getElementById("dropzone");
const inputFoto     = document.getElementById("input-foto");
const preview       = document.getElementById("preview");
const previewVideo  = document.getElementById("preview-video");
let arquivoSelecionado = null;
let tipoSelecionado    = null;   // "foto" | "video"

dropzone.addEventListener("click", () => inputFoto.click());
dropzone.addEventListener("dragover", e => {
  e.preventDefault(); dropzone.classList.add("arrastando");
});
dropzone.addEventListener("dragleave", () => dropzone.classList.remove("arrastando"));
dropzone.addEventListener("drop", e => {
  e.preventDefault(); dropzone.classList.remove("arrastando");
  if (e.dataTransfer.files.length) selecionarArquivo(e.dataTransfer.files[0]);
});
inputFoto.addEventListener("change", e => {
  if (e.target.files.length) selecionarArquivo(e.target.files[0]);
});

function selecionarArquivo(f) {
  arquivoSelecionado = f;
  tipoSelecionado    = (f.type || "").startsWith("video") ? "video" : "foto";
  const reader = new FileReader();
  reader.onload = ev => {
    if (tipoSelecionado === "video") {
      previewVideo.src = ev.target.result;
      previewVideo.hidden = false;
      preview.hidden = true;
    } else {
      preview.src = ev.target.result;
      preview.hidden = false;
      previewVideo.hidden = true;
      previewVideo.removeAttribute("src");
    }
  };
  reader.readAsDataURL(f);
  document.getElementById("btn-processar").disabled = false;
}

document.getElementById("form-ingerir").addEventListener("submit", async e => {
  e.preventDefault();
  const locId = document.getElementById("select-localizacao").value;
  if (!locId || !arquivoSelecionado) return;

  const fd = new FormData();
  fd.append("localizacao_id", locId);
  fd.append("arquivo", arquivoSelecionado);
  document.getElementById("btn-processar").disabled = true;
  document.getElementById("status-ingestao").textContent =
    tipoSelecionado === "video" ? "Enviando vídeo…" : "Enviando foto…";
  document.getElementById("objetos-detectados").innerHTML = "";

  const endpoint = tipoSelecionado === "video"
    ? `${API}/videos/ingerir`
    : `${API}/fotos/ingerir`;

  try {
    const res = await fetchJSON(endpoint, { method: "POST", body: fd });
    if (tipoSelecionado === "video") pollStatusVideo(res.video_id);
    else                              pollStatusFoto(res.foto_id);
  } catch (err) {
    document.getElementById("status-ingestao").textContent = "Erro: " + err.message;
    document.getElementById("btn-processar").disabled = false;
  }
});

async function pollStatusFoto(fotoId) {
  const status = document.getElementById("status-ingestao");
  status.textContent = `Foto #${fotoId}: processando…`;
  while (true) {
    await new Promise(r => setTimeout(r, 3000));
    try {
      const info = await fetchJSON(`${API}/fotos/${fotoId}/status`);
      status.textContent = `Foto #${fotoId}: ${info.status}` +
        (info.objetos_encontrados ? ` — ${info.objetos_encontrados} objeto(s)` : "");
      if (info.status === "concluido") {
        renderObjetosDetectados(info.objetos || []);
        document.getElementById("btn-processar").disabled = false;
        carregarInventario(true);
        return;
      }
      if (info.status === "erro") {
        status.textContent = `Erro: ${info.erro_mensagem || "desconhecido"}`;
        document.getElementById("btn-processar").disabled = false;
        return;
      }
    } catch (e) {
      status.textContent = "Erro ao consultar status: " + e.message;
      document.getElementById("btn-processar").disabled = false;
      return;
    }
  }
}

async function pollStatusVideo(videoId) {
  const status = document.getElementById("status-ingestao");
  status.textContent = `Vídeo #${videoId}: extraindo keyframes…`;
  while (true) {
    await new Promise(r => setTimeout(r, 3000));
    try {
      const info = await fetchJSON(`${API}/videos/${videoId}/status`);
      const tot  = info.frames_extraidos   || 0;
      const proc = info.frames_processados || 0;
      let msg = `Vídeo #${videoId}: ${info.status}`;
      if (tot)  msg += ` — frame ${proc}/${tot}`;
      if (info.objetos_encontrados) msg += ` · ${info.objetos_encontrados} objeto(s)`;
      status.textContent = msg;
      if (info.status === "concluido") {
        renderObjetosDetectados(info.objetos || []);
        document.getElementById("btn-processar").disabled = false;
        carregarInventario(true);
        return;
      }
      if (info.status === "erro") {
        status.textContent = `Erro: ${info.erro_mensagem || "desconhecido"}`;
        document.getElementById("btn-processar").disabled = false;
        return;
      }
    } catch (e) {
      status.textContent = "Erro ao consultar status: " + e.message;
      document.getElementById("btn-processar").disabled = false;
      return;
    }
  }
}

function renderObjetosDetectados(objs) {
  const grid = document.getElementById("objetos-detectados");
  grid.innerHTML = "";
  for (const o of objs) {
    const card = document.createElement("div");
    card.className = "card-objeto";
    const img = o.icone_path ? `/storage/icones/${o.icone_path.split("/").pop()}` : "";
    card.innerHTML = `
      ${img ? `<img src="${img}" alt="${o.nome}" />` : ""}
      <div class="nome">${o.nome}</div>
      <div class="meta">conf ${(o.confianca || 0).toFixed(2)} <span class="badge-fonte">${badgeFonte(o.icone_fonte)}</span></div>
    `;
    card.addEventListener("click", () => abrirModalObjeto(o.id));
    grid.appendChild(card);
  }
}

// ─── Inventário ──────────────────────────────────────────────────────────────
async function carregarInventario(_silencioso) {
  const params = new URLSearchParams();
  const busca       = document.getElementById("filtro-busca").value.trim();
  const categoria   = document.getElementById("filtro-categoria").value;
  const localizacao = document.getElementById("filtro-localizacao").value;
  const estado      = document.getElementById("filtro-estado").value;
  if (busca) params.set("busca", busca);
  if (categoria) params.set("categoria", categoria);
  if (localizacao) params.set("localizacao", localizacao);
  if (estado) params.set("estado", estado);

  try {
    const objs = await fetchJSON(`${API}/objetos?${params.toString()}`);
    const grid = document.getElementById("grid-inventario");
    grid.innerHTML = "";
    if (!objs.length) {
      grid.innerHTML = "<p>Nenhum objeto encontrado.</p>";
      return;
    }
    for (const o of objs) {
      const card = document.createElement("div");
      card.className = "card-objeto";
      const img = o.icone_path ? `/storage/icones/${o.icone_path.split("/").pop()}` : "";
      card.innerHTML = `
        ${img ? `<img src="${img}" alt="${o.nome}" />` : ""}
        <div class="nome">${o.nome}</div>
        <div class="meta">${o.localizacao_nome || "—"}</div>
        <div class="meta"><span class="badge-fonte">${badgeFonte(o.icone_fonte)}</span></div>
      `;
      card.addEventListener("click", () => abrirModalObjeto(o.id));
      grid.appendChild(card);
    }
    // popular filtro de categorias com base nos resultados (uma vez basta)
    if (!document.getElementById("filtro-categoria").dataset.populado) {
      const map = new Map();
      for (const o of objs) if (o.categoria_id) map.set(o.categoria_id, o.categoria_nome);
      const sel = document.getElementById("filtro-categoria");
      for (const [id, nome] of map) sel.innerHTML += `<option value="${id}">${nome}</option>`;
      if (map.size) sel.dataset.populado = "1";
    }
  } catch (err) {
    console.error(err);
  }
}

document.getElementById("btn-recarregar").addEventListener("click", () => carregarInventario());
document.getElementById("filtro-busca").addEventListener("keydown", e => {
  if (e.key === "Enter") carregarInventario();
});

// ─── Modal de detalhes ──────────────────────────────────────────────────────
async function abrirModalObjeto(id) {
  try {
    const o = await fetchJSON(`${API}/objetos/${id}`);
    const corpo = document.getElementById("modal-corpo");
    const img = o.icone_path ? `/storage/icones/${o.icone_path.split("/").pop()}` : "";
    corpo.innerHTML = `
      <h3>${o.nome}</h3>
      ${img ? `<img src="${img}" alt="${o.nome}" />` : ""}
      <dl>
        <dt>Categoria</dt><dd>${o.categoria_nome || "—"}</dd>
        <dt>Localização</dt><dd>${o.localizacao_nome || "—"}${o.localizacao_comodo ? " (" + o.localizacao_comodo + ")" : ""}</dd>
        <dt>Cor</dt><dd>${o.cor || "—"}</dd>
        <dt>Tamanho</dt><dd>${o.tamanho || "—"}${o.tamanho_estimado_cm ? " · " + o.tamanho_estimado_cm : ""}</dd>
        <dt>Material</dt><dd>${o.material || "—"}</dd>
        <dt>Estado</dt><dd>${o.estado || "—"}</dd>
        <dt>Função</dt><dd>${o.funcao || "—"}</dd>
        <dt>Descrição</dt><dd>${o.descricao || "—"}</dd>
        <dt>Confiança</dt><dd>${(o.confianca || 0).toFixed(2)} (${o.modelo_visao || "?"})</dd>
        <dt>Fonte do ícone</dt><dd>${badgeFonte(o.icone_fonte)}</dd>
      </dl>
      <h4>Editar</h4>
      <form id="form-editar">
        <input type="text" name="nome" value="${o.nome || ""}" placeholder="Nome" />
        <input type="text" name="cor" value="${o.cor || ""}" placeholder="Cor" />
        <input type="text" name="material" value="${o.material || ""}" placeholder="Material" />
        <input type="text" name="estado" value="${o.estado || ""}" placeholder="Estado" />
        <input type="text" name="funcao" value="${o.funcao || ""}" placeholder="Função" />
        <button type="submit">Salvar</button>
        <button type="button" id="btn-excluir" style="background:#c0392b">Excluir</button>
      </form>
    `;
    document.getElementById("modal").hidden = false;
    document.getElementById("form-editar").addEventListener("submit", async e => {
      e.preventDefault();
      const data = Object.fromEntries(new FormData(e.target));
      await fetchJSON(`${API}/objetos/${id}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data),
      });
      document.getElementById("modal").hidden = true;
      carregarInventario();
    });
    document.getElementById("btn-excluir").addEventListener("click", async () => {
      if (!confirm("Excluir este objeto?")) return;
      await fetchJSON(`${API}/objetos/${id}`, { method: "DELETE" });
      document.getElementById("modal").hidden = true;
      carregarInventario();
    });
  } catch (e) {
    alert("Erro: " + e.message);
  }
}
document.getElementById("modal-fechar").addEventListener("click", () => {
  document.getElementById("modal").hidden = true;
});

// ─── Assistente (chat) ───────────────────────────────────────────────────────
document.querySelectorAll(".chip").forEach(c => {
  c.addEventListener("click", () => {
    document.getElementById("input-pergunta").value = c.textContent;
    document.getElementById("form-chat").requestSubmit();
  });
});

document.getElementById("form-chat").addEventListener("submit", async e => {
  e.preventDefault();
  const inp = document.getElementById("input-pergunta");
  const pergunta = inp.value.trim();
  if (!pergunta) return;
  adicionarBalao("usuario", pergunta);
  inp.value = "";
  adicionarBalao("assistente", "Pensando…", "ollama", "balao-temp");
  try {
    const r = await fetchJSON(`${API}/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ pergunta }),
    });
    document.querySelectorAll(".balao-temp").forEach(n => n.remove());
    adicionarBalao("assistente", r.resposta, r.modelo);
  } catch (err) {
    document.querySelectorAll(".balao-temp").forEach(n => n.remove());
    adicionarBalao("assistente", "Erro: " + err.message, "ollama");
  }
});

function adicionarBalao(quem, texto, modelo, classeExtra = "") {
  const div = document.createElement("div");
  div.className = `balao ${quem} ${classeExtra}`;
  if (quem === "assistente") {
    div.innerHTML = `${badgeModelo(modelo)} ${texto}`;
  } else {
    div.textContent = texto;
  }
  document.getElementById("chat-mensagens").appendChild(div);
  div.scrollIntoView({ behavior: "smooth", block: "end" });
}

// ─── Boot ────────────────────────────────────────────────────────────────────
(async function init() {
  await carregarModo();
  await carregarLocalizacoes(true);
  await carregarInventario();
})();
