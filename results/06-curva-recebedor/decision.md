# Decisão — Frente 06: curva do recebedor (run 20260930-0720)

## Atribuição (R0)

Recebedor e defensor primários atribuídos em **10.871/10.871** cruzamentos (jogador de
ataque — cruzador excluído — e defensor mais próximos da bola no último instante da
janela). Distância mediana do recebedor à bola na chegada: **3,08 m** — atribuição
sensata (o "alvo" real do cruzamento).

## A quem pertence o bloco chegada (estabilidade temporal, xgboost, ≥20 disputas)

| Papel | stab(arrival) | stab(taxa bruta) | n |
|---|---|---|---|
| **Recebedor** | **0,380** | 0,255 | 138 |
| **Defensor primário** | **0,356** | 0,219 | 127 |
| Cruzador (controle) | 0,216 | −0,026 | 155 |

**s1 = True (delta recebedor−cruzador = +0,165; critério ≥ +0,10).**

## Leitura

1. **O bloco chegada pertence mais a quem disputa a bola do que a quem a cruza.**
   Recebedor (0,38) e defensor (0,36) superam o cruzador (0,22) em ~0,16 — o contest
   aéreo é propriedade parcial dos disputantes.
2. **Mas continua sendo a parte ruidosa.** 0,38 está muito abaixo da situação do toque
   (0,53-0,74): o duelo aéreo é o elo mais estocástico da cadeia — mais estável que o
   cruzador, menos estável que qualquer sinal individual forte.
3. **Estar no fim de cruzamentos certos é levemente estável** (taxa bruta por recebedor
   0,255 vs −0,03 do cruzador): "quem ataca a área boa" tem persistência — seleção de
   alvo dos companheiros + posicionamento — mas fraca.
4. **Validação interna:** crosser_ctrl/arrival = 0,21551 = frente 03 exata (mesma
   régua, mesma amostra) — a maquinaria de atribuição está consistente.

## O arco do crédito completo (todas as frentes)

| Papel | O que controla (estável) | Força |
|---|---|---|
| Cruzador | a situação de onde cruza | 0,53–0,74 |
| Bola (voo) | nada individual | 0,09–0,12 |
| Recebedor/Defensor | o duelo na chegada | 0,36–0,38 |
| Outcome (taxa bruta) | ninguém | −0,03 |

**"O cruzador escolhe, a bola voa, o recebedor disputa — e o desfecho é obra de todos
e de ninguém."** A cadeia de crédito do cruzamento mapeada ponta a ponta.

## Limitações

- Loop bug: só xgboost foi registrado (adaboost calculado e sobrescrito) — o padrão
  da família xgboost é o reportado em todas as frentes; regenerar adaboost fica para o
  paper completo.
- Recebedor por proximidade ≠ recebedor intencional (o alvo real pode ser outro);
  mediana 3,1 m sugere acerto alto, mas é proxy.
- Defensor primário ignora marcação por zona vs individual; GK não separado.
- n=138/127 jogadores qualificados: poder moderado; ICs por bootstrap de jogadores no
  paper completo.

## Decisão: **supported** — o "QUEM" agora tem três papéis, não um
