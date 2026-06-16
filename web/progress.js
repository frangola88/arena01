/**
 * ProcessingProgress — Componente para mostrar progresso de foto/vídeo
 * Polling a cada 500ms, exibe: etapa + descrição + % + tempo
 */

class ProcessingProgress {
  constructor(fotoId, containerId, onComplete = null, onCancel = null) {
    this.fotoId = fotoId;
    this.container = document.getElementById(containerId);
    this.onComplete = onComplete;
    this.onCancel = onCancel;
    this.polling = null;
    this.render();
    this.startPolling();
  }

  render() {
    this.container.innerHTML = `
      <div class="progress-container">
        <div class="progress-description">
          <div class="etapa-icon">⏳</div>
          <div class="description-text">
            <p class="etapa-nome">Iniciando...</p>
            <p class="descricao-detalhe">Preparando processamento</p>
          </div>
        </div>
        <div class="progress-bar-container">
          <div class="progress-bar">
            <div class="progress-fill" style="width: 0%"></div>
          </div>
          <div class="progress-label">0% (0/0 objetos)</div>
        </div>
        <div class="progress-timing">
          <span class="tempo-decorrido">⏱️ 0s</span>
          <span class="tempo-restante">⏳ ... faltando</span>
        </div>
        <div class="progress-actions" style="display: none;">
          <button class="btn-cancelar">⛔ Cancelar Processamento</button>
        </div>
      </div>
    `;

    // Anexar listener ao botão cancelar
    const btnCancelar = this.container.querySelector('.btn-cancelar');
    if (btnCancelar) {
      btnCancelar.addEventListener('click', () => this.handleCancelar());
    }
  }

  async startPolling() {
    this.polling = setInterval(() => this.updateProgress(), 500);
    // Primeira atualização imediata
    this.updateProgress();
  }

  async updateProgress() {
    try {
      const res = await fetch(`/api/fotos/${this.fotoId}/progresso`);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      
      const data = await res.json();
      this.renderProgress(data);

      // Assim que sai da segmentação, o debug.jpg já existe — mostra link cedo
      if (data.etapa_atual && data.etapa_atual !== 'iniciando' && data.etapa_atual !== 'segmentando') {
        this._mostrarLinkDebug();
      }

      // Se concluído ou erro, parar polling
      if (data.etapa_atual === 'concluído' || data.etapa_atual === 'erro' || data.percentual === 100) {
        clearInterval(this.polling);
        this._mostrarLinkDebug();
        if (data.etapa_atual === 'concluído' && this.onComplete) {
          setTimeout(this.onComplete, 1500);
        }
      }
    } catch (err) {
      console.error('Erro ao buscar progresso:', err);
      // Continuar tentando
    }
  }

  renderProgress(data) {
    const fill = this.container.querySelector('.progress-fill');
    const label = this.container.querySelector('.progress-label');
    const etapaNome = this.container.querySelector('.etapa-nome');
    const descricao = this.container.querySelector('.descricao-detalhe');
    const icon = this.container.querySelector('.etapa-icon');
    const tempoDecorrido = this.container.querySelector('.tempo-decorrido');
    const tempoRestante = this.container.querySelector('.tempo-restante');
    const actions = this.container.querySelector('.progress-actions');
    const btnCancelar = this.container.querySelector('.btn-cancelar');

    // Ícone e nome da etapa
    const icons = {
      'segmentando': '✂️',
      'analisando': '🔍',
      'enriquecendo': '✨',
      'finalizando': '📦',
      'concluído': '✅',
      'erro': '❌'
    };
    const nomes = {
      'segmentando': 'Segmentando objetos',
      'analisando': 'Analisando detalhes',
      'enriquecendo': 'Enriquecendo dados',
      'finalizando': 'Finalizando',
      'concluído': 'Concluído!',
      'erro': 'Erro no processamento'
    };

    icon.textContent = icons[data.etapa_atual] || '⏳';
    etapaNome.textContent = nomes[data.etapa_atual] || 'Processando...';
    descricao.textContent = data.erro_mensagem || data.descricao_etapa || 'Processando...';

    // Barra de progresso
    fill.style.width = `${data.percentual}%`;
    label.textContent = `${data.percentual}% (${data.objetos_processados}/${data.objetos_totais} objetos)`;

    // Tempo
    tempoDecorrido.textContent = `⏱️ ${this.formatarTempo(data.tempo_decorrido_s)}`;
    if (data.tempo_restante_s !== null && data.tempo_restante_s !== undefined) {
      tempoRestante.textContent = `⏳ ~${this.formatarTempo(data.tempo_restante_s)} restantes`;
    } else {
      // ETA ainda não é confiável — não mostrar números falsos
      tempoRestante.textContent = '⏳ calculando…';
    }

    // Mostrar botão cancelar apenas se processando
    const podeCanc = data.etapa_atual !== 'erro' && data.etapa_atual !== 'concluído';
    if (podeCanc) {
      actions.style.display = 'flex';
      btnCancelar.disabled = false;
      btnCancelar.textContent = '⛔ Cancelar Processamento';
    } else {
      actions.style.display = 'none';
    }

    // Mostrar erro se houver
    let errorDiv = this.container.querySelector('.progress-error');
    if (data.erro_mensagem && data.etapa_atual === 'erro') {
      if (!errorDiv) {
        errorDiv = document.createElement('div');
        errorDiv.className = 'progress-error';
        this.container.querySelector('.progress-container').appendChild(errorDiv);
      }
      errorDiv.textContent = `⚠️ ${data.erro_mensagem}`;
    } else if (errorDiv) {
      errorDiv.remove();
    }
  }

  async handleCancelar() {
    if (!confirm('Tem certeza que quer cancelar o processamento?')) {
      return;
    }

    const btnCancelar = this.container.querySelector('.btn-cancelar');
    btnCancelar.disabled = true;
    btnCancelar.textContent = '⏳ Cancelando...';

    try {
      const res = await fetch(`/api/fotos/${this.fotoId}/cancelar`, {
        method: 'POST'
      });

      if (res.ok) {
        const data = await res.json();
        console.log(data.mensagem);
        
        // Parar polling
        clearInterval(this.polling);
        
        // Atualizar status
        const statusData = {
          etapa_atual: 'erro',
          descricao_etapa: '',
          erro_mensagem: 'Cancelado pelo usuário',
          percentual: 0,
          tempo_decorrido_s: 0,
          tempo_restante_s: null,
          objetos_processados: 0,
          objetos_totais: 1
        };
        this.renderProgress(statusData);
        
        if (this.onCancel) this.onCancel();
      } else {
        alert('Erro ao cancelar: ' + res.statusText);
        btnCancelar.disabled = false;
        btnCancelar.textContent = '⛔ Cancelar Processamento';
      }
    } catch (err) {
      console.error('Erro ao cancelar:', err);
      alert('Erro ao cancelar processamento');
      btnCancelar.disabled = false;
      btnCancelar.textContent = '⛔ Cancelar Processamento';
    }
  }

  _mostrarLinkDebug() {
    // Adiciona um botão "Ver detecção visual (debug)" abaixo da barra de progresso.
    if (this.container.querySelector('.btn-debug-visual')) return; // já adicionado

    const link = document.createElement('a');
    link.href = `/api/fotos/${this.fotoId}/debug`;
    link.target = '_blank';
    link.className = 'btn-debug-visual';
    link.textContent = '🔍 Ver detecção visual (debug)';
    link.title = 'Mostra os contornos detectados pelo OpenCV: verde = escolhidos, vermelho = descartados por grandes, cinza = descartados por pequenos.';

    const actions = this.container.querySelector('.progress-actions');
    if (actions) {
      actions.style.display = 'flex';
      actions.appendChild(link);
    }
  }

  formatarTempo(segundos) {
    if (segundos === null || segundos === undefined) return '...';
    segundos = Math.max(0, Math.floor(segundos));
    if (segundos < 60) return `${segundos}s`;
    const m = Math.floor(segundos / 60);
    const s = segundos % 60;
    if (m < 60) return s > 0 ? `${m}min ${s}s` : `${m}min`;
    const h = Math.floor(m / 60);
    const mr = m % 60;
    return `${h}h ${mr}min`;
  }

  destroy() {
    if (this.polling) clearInterval(this.polling);
  }
}

// Exportar para uso global
window.ProcessingProgress = ProcessingProgress;
