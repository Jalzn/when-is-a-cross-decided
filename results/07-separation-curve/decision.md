# Decisão — Frente 07: curva da separação (run 20260930-0820)

## O achado central (mecanismo)

**Trajetória média da separação recebedor–defensor, por desfecho (10.871 cruzamentos):**

| fração do voo | sucesso | fracasso | gap |
|---|---|---|---|
| 0,1 | 4,57 m | 6,63 m | **2,06 m** |
| 0,5 | 3,57 m | 5,38 m | 1,81 m |
| 1,0 | 3,02 m | 5,05 m | 2,03 m |

**O duelo aéreo dos cruzamentos bem-sucedidos é ~2 m mais fechado desde o primeiro
décimo do voo — e o gap persiste.** No cruzamento que dá certo, o recebedor disputa
corpo-a-corpo (marcação apertada, atacante vence a posição); no que falha, o atacante
mais próximo está longe do defensor (a bola cai em espaço disputado por segundo, ou o
defensor antecipa). A "primeira divergência ≥ 0,3 m" ocorre em f = 0,1 — **o destino do
duelo está desenhado no começo do voo**, coerente com a curva de informação (2/3 da
decisão no primeiro décimo) e com o bloco chegada como carregador único.

## Poder preditivo-descritivo

- **sep sozinha no f100: AUC 0,879** — a feature de duelo mais forte que qualquer bloco
  anterior (arrival 0,824; fields 0,797) com só 10 variáveis.
- **união + sep no f100: 0,897 vs 0,818** — +0,080, o maior ganho incremental medido
  nesta linha.

## ⚠️ Vazamento por seleção de identidade (limitação séria, declarada)

Recebedor/defensor são identificados pela proximidade à bola **na chegada** — a escolha
de QEM rastrear usa informação futura correlacionada ao desfecho. Consequências:
- Os valores PRÉ-TOQUE da curva sep (0,69–0,75) **não são claims preditivos** — são
  descritivos do contest eventual (o mesmo estatuto epistêmico do bloco chegada, mas a
  contaminação é maior porque afeta a identidade dos jogadores rastreados).
- Os cortes pós-toque e a figura de mecanismo são análise a posteriori legítima (o mesmo
  enquadramento do paper: avaliação de entrega e atribuição, não previsão).
- Para o paper completo: re-identificar recebedor sem informação futura (mais próximo
  da bola no instante corrente, dinâmico) e replicar.

## Decisão: **supported (como mecanismo a posteriori; sem claim preditivo pré-toque)**

O arco mecanístico fecha: a bola revela o destino nos primeiros 10% do voo (f05) e o
**duelo que decide já está separado nesse mesmo instante** (f07) — bola e contest são as
duas faces da decisão precoce, e a chegada é onde elas se consumam.
