# Pre-commit Hook — Validação Local de Testes

Garante que testes passem antes de qualquer commit, reforcando qualidade localmente.

---

## 🚀 Setup (primeira vez)

O hook já está instalado em `.git/hooks/pre-commit`. Para ativar (se necessário):

```bash
chmod +x .git/hooks/pre-commit
```

Ou usar o script de setup:
```bash
./scripts/setup_precommit.sh
```

---

## 🔄 Funcionamento

A cada tentativa de `git commit`:

1. **Pre-commit hook dispara automaticamente**
   ```bash
   $ git commit -m "fix: bug na segurança"
   [pre-commit] Validando commit com pytest...
   [pre-commit] Executando 252 testes (~1.6s)...
   ```

2. **Pytest roda (252 testes)**
   - Se **todos passarem** → commit é criado ✓
   - Se **algum falhar** → commit é bloqueado ✗

3. **Resultado:**
   ```bash
   # ✓ Sucesso
   [pre-commit] ✓ Todos os testes passaram
   [main d39bfd3] fix: bug na segurança
   
   # ✗ Falha
   [pre-commit] ERRO: testes falharam
   
   Opções:
     1. Corrigir o código e tentar novamente
     2. git commit --no-verify (bypass para emergências)
     3. Rodar com debug: pytest -xvs
   ```

---

## ⏱️ Tempo de execução

- **Normal:** ~1.6 segundos (252 testes)
- **Timeout:** 60 segundos (segurança contra travamentos)

---

## 🚨 Emergências — Bypass

Se precisar commitar sem rodar testes (RARAMENTE):

```bash
git commit --no-verify -m "emergência: fix temporário"
```

**Aviso:** isto contorna a validação. Revert quando possível.

---

## 🔧 Troubleshooting

### Hook não dispara
```bash
# Verificar permissões
ls -la .git/hooks/pre-commit
chmod +x .git/hooks/pre-commit

# Verificar se conda funciona
source /home/cuco/miniconda/etc/profile.d/conda.sh && conda activate casaiq && pytest --version
```

### Pytest não encontrado
```bash
# Ativar ambiente e instalar
mamba activate casaiq
pip install -r requirements.txt
```

### Hook trava/timeout
```bash
# Aumentar timeout em .git/hooks/pre-commit (linha MAX_TIME=...)
# Ou rodar testes manualmente para diagnosticar:
pytest -xvs
```

### Remover hook temporariamente
```bash
rm .git/hooks/pre-commit
# Reinstalar depois:
chmod +x .git/hooks/pre-commit  # ou ./scripts/setup_precommit.sh
```

---

## 📊 Impacto na workflow

| Ação | Tempo | Impacto |
|---|---|---|
| `git commit` (com hook) | +1.6s | Garante qualidade |
| `git commit --no-verify` | 0s | Bypass para emergências |
| CI workflow (GitHub Actions) | ~30s | Validação na cloud |

---

## 🎯 Boas práticas

✅ **Commit com frequência** — o hook valida rápido  
✅ **Escrever testes junto com código** — hook detecta regressões imediatamente  
✅ **Usar `--no-verify` raramente** — só para emergências documentadas  
✅ **Rodar `pytest` localmente antes de commitar** — detecção precoce

---

## 🔗 Relacionado

- [README.md](README.md) — Instalação, estrutura, troubleshooting geral
- [STATUS.md](STATUS.md) — Progresso do projeto, pendências
- `.github/workflows/test.yml` — CI na cloud (redundância)

---

*Documentação: 2026-05-13*  
*Pendência #8 — Pre-commit hook*
