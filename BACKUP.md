# CasaIQ — Backup e Restore de DB

Guia de uso dos scripts de backup automático do banco de dados SQLite.

---

## 🤖 Backup Automático via Cron

### Setup (primeira vez)

1. **Teste manual** para garantir que funciona:
   ```bash
   ./scripts/backup_db.sh
   ```
   Deve criar um arquivo em `.backups/casaiq_YYYY-MM-DD_HHmmss.db`.

2. **Ativar cron job** — execute:
   ```bash
   crontab -e
   ```
   Adicione uma das linhas abaixo (escolha horário que funciona melhor):

   **Daily às 2am** (recomendado):
   ```
   0 2 * * * /home/cuco/projetos/casaiq/scripts/backup_db.sh >> /tmp/casaiq_backup.log 2>&1
   ```

   **Cada 6 horas**:
   ```
   0 */6 * * * /home/cuco/projetos/casaiq/scripts/backup_db.sh >> /tmp/casaiq_backup.log 2>&1
   ```

   **Cada hora**:
   ```
   0 * * * * /home/cuco/projetos/casaiq/scripts/backup_db.sh >> /tmp/casaiq_backup.log 2>&1
   ```

3. **Verificar status**:
   ```bash
   tail -f /tmp/casaiq_backup.log
   ```

### Retenção automática

- Backups são mantidos pelos **últimos 30 dias**
- Diariamente, backups com >30 dias são deletados
- Logs incluem: timestamp, arquivo criado, tamanho, e contagem total

---

## 📁 Estrutura de Backups

```
casaiq/
├── casaiq.db           ← banco de produção (sempre atual)
├── .backups/           ← pasta de backups (versionada via .gitignore)
│   ├── casaiq_2026-05-13_021500.db
│   ├── casaiq_2026-05-12_021234.db
│   └── casaiq_2026-05-11_020145.db
├── scripts/
│   ├── backup_db.sh    ← cria backup + limpa antigos
│   └── restore_db.sh   ← restaura a partir de backup
```

---

## 🔄 Restaurar de um Backup

### Restauração simples

```bash
./scripts/restore_db.sh .backups/casaiq_2026-05-13_021500.db
```

O script:
1. Verifica que o arquivo existe
2. Faz um backup de segurança do DB atual (`.restore_safety_*.bak`)
3. Copia o backup para `casaiq.db`
4. Reporta sucesso ou reverte em caso de erro

### Listar backups disponíveis

```bash
ls -lh .backups/
```

### Restauração manual (sem script)

Se o script falhar, você pode restaurar manualmente:

```bash
# Opção A: Cópia simples (mais rápido, sem verificação SQLite)
cp .backups/casaiq_2026-05-13_021500.db casaiq.db

# Opção B: Via sqlite3 (mais seguro, verifica integridade)
sqlite3 .backups/casaiq_2026-05-13_021500.db ".backup 'casaiq.db'"
```

---

## ⚠️ Boas práticas

- **Não deletar `.backups/` manualmente** — deixa que o cron cuide da retenção
- **Fazer teste de restore periodicamente** — garante que os backups são válidos:
  ```bash
  ./scripts/backup_db.sh && \
  cp casaiq.db casaiq.db.test && \
  ./scripts/restore_db.sh .backups/casaiq_$(date +%Y-%m-%d)*.db && \
  cp casaiq.db.test casaiq.db
  ```
- **Monitorar logs** — qualquer erro no cron aparece em `/tmp/casaiq_backup.log`

---

## 🐛 Troubleshooting

### "sqlite3: comando não encontrado"
Instale o SQLite:
```bash
sudo zypper install sqlite3
```

### "Permissão negada ao criar backup"
Verifique permissões:
```bash
ls -la casaiq.db scripts/
# casaiq.db deve estar legível, scripts/ deve estar executável
chmod 755 scripts/backup_db.sh scripts/restore_db.sh
```

### "DB corrompido após restore"
Se a restauração deixar o DB em estado inválido:
```bash
# Recuperar do backup de segurança
cp casaiq.db.restore_safety_*.bak casaiq.db
# Contatar desenvolvedor ou restaurar de backup anterior
```

---

## 📊 Monitoramento

Adicionar ao seu arquivo de observabilidade:

```bash
# Verificar tamanho total de backups
du -sh .backups/

# Contar backups
ls -1 .backups/ | wc -l

# Backup mais antigo
ls -t .backups/ | tail -1

# Tamanho do banco principal
du -h casaiq.db
```

---

*Documentação atualizada em 2026-05-13.*
