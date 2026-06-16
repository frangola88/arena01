# Investigação: separar objetos finos próximos na tomografia de cena (2026-06-16)

Contexto: `core/analise_cena.py` subcontava objetos em cenas densas de
ferramentas finas alinhadas (dava 8/7 onde a verdade era 14). Fotos de
referência: `data/fotos_mestras/IMG_mestra04{sem,com}flash.jpg` (mesma cena,
14 objetos sob a regra "embalagem fechada = 1").

## O que foi testado (em ordem) e resultado

| # | Hipótese | Protótipo | Resultado |
|---|----------|-----------|-----------|
| 1 | Grids rotacionados ±45° recuperam separação | `demo_analise_cena_rotada.py` | **Nulo.** Gaussian σ=4 é isotrópico — rotacionar mapa já suavizado não acrescenta nada. 8/8/8 em todos os eixos. |
| 2 | Aumentar resolução (block 16→8, σ menor) | `demo_analise_cena_multipass.py` | **Pior.** Variância pura conta grão de madeira: explodiu pra 63/48. Refino local multiplicou ruído (795/900). |
| 3 | Trocar variância por **coerência de gradiente** (structure tensor) | `demo_analise_cena_coerencia.py` | **Funciona.** Coerência separa borda real de textura. 12-13 com th=70. Sinal nítido por ferramenta. |
| 4 | Malha 100×100 + Otsu + componentes conexos (sem griddata) | `demo_grid_celulas.py` | 13/11. Precisa `closing=2` p/ preencher interior oco das ferramentas (coerência só acende na borda). Otsu global sofre com iluminação desigual (flash funde fileira via borda do tapete). |
| 5 | Threshold adaptativo local em vez de Otsu | (sweep) | **Pior** (6/8). O merge do flash vem do `closing` fazer ponte sobre a borda linear do tapete — distinção semântica, não de threshold. |
| 6 | Overlay (2ª imagem com regiões numeradas) ajuda a contagem do Claude? | `teste_overlay_api.py` + `teste_overlay_api_regra.py` | Ver abaixo. |

## Teste A/B/C real contra a API (verdade=14)

| condição | semflash | comflash | erro médio |
|---|---|---|---|
| A só foto | 25 | 28 | 12.5 |
| B +hint numérico | 22 | 26 | 10 |
| A' +regra "fechada=1" no texto | 21 | 22 | 7.5 |
| B' +regra +hint | 18 | 14 | 2.0 |
| C +overlay | 13 | 16 | 1.5 |

**Conclusão do overlay:** o ganho grande vem de **declarar a regra de contagem
no texto** (erro 12→2), que o prompt de produção (`core/skills/inventario_visao.md`
linha 15a) **já tem**. O overlay adiciona só margem mínima (2.0→1.5), empatada
com o ruído de 2 amostras, ao custo de 2ª imagem + pipeline malha/overlay.
**Não integrado.** Reavaliar só com cenas densas de verdade (30+ objetos), onde
a contagem espontânea do modelo pode desmoronar e a orientação estrutural
passar a importar.

## O que FOI para produção

`core/analise_cena.py`: sinal trocado de variância → **coerência de gradiente**
(`_coerencia_local`), σ_pré 4→1, TH_OBJ_PCT 65→70. Hint subiu de 8/7 → 10/12
(verdade 14). 302 testes passando. `_variancia_local` mantido p/ compat/testes.

## O que NÃO foi (arquivado como POC)

Malha 100×100, overlay, grids rotacionados/multipass. Decisão de domínio que
ancorou tudo: **embalagem fechada = 1 objeto**.
