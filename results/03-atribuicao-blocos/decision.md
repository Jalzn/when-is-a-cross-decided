# Decisão — Frentes 03+04: atribuição por blocos e skill por bloco

**Run:** 20260930-0410 na h100-2 (job-5262c4ba; 2 tentativas — bug da coluna categórica
`cross_region` corrigido). Local: runs/1/. Corte f100 (chegada), 10.871 cruzamentos, 155
cruzadores qualificados (≥20).

## O QUÊ — atribuição por bloco (xgboost/success, OOF)

| Bloco | AUC isolado | ΔAUC leave-one-out [IC 95%] | Único? |
|---|---|---|---|
| strike (toque, 25 feats) | 0,554 | +0,0001 [−0,001; +0,001] | não — redundante |
| fields_flight (área no voo, 90) | 0,797 | +0,0042 [+0,001; +0,007] | sim (pequeno) |
| ball2d (bola 2D, 7) | 0,773 | +0,0002 [−0,001; +0,001] | não — subsumida |
| **flight3d (técnica do voo, 13)** | 0,761 | **+0,0159 [+0,013; +0,019]** | **sim (maior)** |
| **arrival (geometria da chegada, 19)** | **0,824** | **+0,0131 [+0,011; +0,016]** | **sim** |
| união (135) | **0,849** | — | — |

Leitura: a informação única que resolve o cruzamento mora na **técnica do voo 3D** e na
**geometria da chegada**; a configuração do toque e a posição 2D da bola são redundantes
dado o resto (são versões parciais dos mesmos sinais). s1 complementaridade ✓ (todos os
isolados < união). s2 como definido ✗ — mas a estrutura de redundância é em si um
achado: os blocos "únicos" são exatamente bola-3D e chegada.

## QUEM — estabilidade temporal por bloco (155 cruzadores, metades cronológicas)

| Bloco (adaboost / xgboost) | Spearman split-half temporal |
|---|---|
| **strike (situação no toque)** | **0,740 / 0,527** |
| fields_flight | 0,350 / 0,195 |
| arrival | 0,264 / 0,216 |
| ball2d | 0,081 / 0,136 |
| flight3d (técnica) | 0,090 / 0,124 |
| união | 0,183 / 0,128 |
| taxa bruta (referência) | **−0,026** |

**s3 refutado na direção da hipótese e invertido:** a técnica do voo NÃO é skill estável
(0,09-0,12, perto do ruído); a chegada também não (0,22-0,26). **O único bloco estável é
a situação que o cruzador cria/ocupacional no instante do toque (0,53-0,74)** — 4× a
6× qualquer bloco do voo, com a taxa bruta replicando o ruído do POC I (−0,026 vs −0,01).

## A história completa (com a frente 02)

1. **QUANDO:** o desfecho é decidido no ar — ¼ do voo carrega 75% do salto total (+0,24).
2. **O QUÊ:** o que resolve é a bola em 3D (técnica de entrega) + o contest na chegada.
3. **QUEM:** nada disso é skill individual mensurável do cruzador. O único sinal estável
   por jogador é a **situação do toque** — seleção de contexto, não execução.
   *Selection over execution* — ecoa o finalista La Pausa (timing/seleção) e estende o
   "outcome is noise" do MLSA: agora decomposto, o ruído está no voo e na chegada.

Nota: união (0,13-0,18) é MENOS estável que strike isolado — fusão com o voo custa
repetibilidade (replica o s4 da frente 07 do xcross-lab).

## Limitações

- Estabilidade por bloco usa o OOF do bloco ISOLADO (não contribuição condicional);
  a chave cronológica é a substituta validada (match_id+frame, frente 10), e as frentes
  11-12 do xcross-lab documentaram sensibilidade da métrica à convenção de corte
  (0,28-0,66 através de amostras) — os números aqui valem como ORDENAÇÃO entre blocos
  sob a mesma régua, não como constantes absolutas.
- flight3d usa 13 features canônicas computadas do voo completo (retrospectivas por
  construção — coerente com o corte f100).
- s2/s3 como previstos falharam; a leitura acima é a reinterpretação honesta —
  resultado negativo da hipótese "técnica é skill" é válido e informativo.

## Decisão: **supported (com hipótese central de skill refutada e invertida)**

Os três eixos do paper estão medidos. O conceito resiste mais forte: a decisão está no
ar, mas o crédito ao cruzador está no que ele faz ANTES do toque.

---

## Adendo (2026-09-30, pós-decisão): blindagem da claim de skill

Ameaça (frente 10 do xcross-lab: preditor posicional atinge 0,56 de estabilidade) testada
com 3 variantes do bloco strike. Resultado (runs/shield/): strike sem posição = **0,718**
(vs 0,740 completo); piso posicional = **0,48**; margem +0,24. **Claim sobrevive** com a
nuance: parte é papel, o resto é qualidade de situação genuína. Resultado r-2014adba;
abstract ajustado.
