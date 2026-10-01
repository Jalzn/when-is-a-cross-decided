# Decisão — Frente 02: curva de informação

**Run:** 20260930-0245 na h100-2 (job-c1e04971; 3 tentativas, 2 correções de código no
caminho — rótulo por join, mkdir do run dir). Local: runs/1/.

## A curva primária (union = campos + bola, XGBoost, success, OOF)

| Corte | AUC | IC 95% |
|---|---|---|
| k−10 (2s antes do toque) | 0,548 | — |
| k−7 | 0,553 | — |
| k−4 | 0,557 | — |
| k−2 | 0,562 | — |
| **k0 (toque)** | **0,578** | — |
| **f25 (¼ do voo)** | **0,758** | — |
| f50 | 0,781 | — |
| f75 | 0,797 | — |
| **f100 (chegada)** | **0,818** | — |

Salto toque→chegada: **+0,240 [0,229; 0,253]** (bootstrap pareado 500) — 2,4× o
critério s2. **O primeiro quarto do voo carrega 75% do salto total** (+0,180).

## Os três achados

1. **"O cruzamento é decidido no ar."** Antes do toque, o desfecho é quase plano
   (0,548→0,578 ao longo de 2 s de aproximação — inclui o próprio toque). No primeiro
   quarto do voo a discriminação salta para 0,758. Não é um degrau no fim (s3 passou):
   a curva sobe cedo e continua subindo.
2. **A bola é o primeiro resolutivo.** A curva ball-only: k0 = 0,521 → f25 = 0,751 →
   f100 = 0,772. A TRAJETÓRIA 2D da bola sozinha (sem z!) no ¼ do voo já dá 0,751 —
   mais cedo que os campos dos jogadores (f25 fields = 0,661). Direção/qualidade da
   entrega são legíveis imediatamente.
3. **Os jogadores decidem a chegada.** Fields-only termina acima (f100: 0,797 vs
   0,772 da bola): a configuração na área resolve o desfecho ao final, superando a
   informação da bola. Complementaridade: union vence em todos os cortes.

## Controles

- **Vazamento de comprimento:** braço D (tempo absoluto + covariáveis de comprimento):
   fields+meta a 1,2s = 0,752 vs fields puro a 1,2s = 0,736 → comprimento soma +0,017,
   mas o ganho dos campos SEM comprimento é +0,16 sobre k0 — o sinal não é o comprimento.
   Cortes por fração (braço primário) são imunes por construção.
- **Consistência interna:** a0 ≡ k0 exato (0,5779/0,5208) — maquinaria de corte ok.
- **Âncora externa:** k0 union = 0,578 = AUC publicada do xCross no POC I (0,578);
   f100 = 0,818 na faixa do xCrossOT+voo da frente 07 (0,827 na âncora retreinada).

## Critérios

s1 monotonicidade ✓ · s2 salto ≥0,10 com IC ✓ (+0,240) · s3 intermediário ≥50% ✓ (75%).

## Decisão: **supported** — a figura-âncora do paper existe e não é trivial

A curva conta a história do paper em um gráfico: aproximação plana → toque adiciona
pouco → o voo resolve, cedo (bola primeiro, área no fim).

## Limitações

- Ball-only usa xy do npz (z não está armazenado) — o papel do voo 3D completo é da frente 03.
- Cortes por fração usam o comprimento DO PRÓPRIO cruzamento para normalizar (definição
  "informação disponível até f% deste voo"); braço D cobre a alternativa absoluta.
- f100 inclui cruzamentos "out" (bola fora ⇒ sucesso 0 conhecido) — legítimo para a
  leitura "na chegada", e o contraste com f25 (que não vê o fim) mostra que o sinal é
  anterior ao encerramento.
- Pré-toque fraco nesta família de features (0,548–0,578): consistente com frente 06
  inconclusiva; a questão "pré-toque na amostra cheia" (frente 13 congelada) permanece
  em aberto com outra família (90 colunas + lr).
