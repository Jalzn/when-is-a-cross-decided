# Decisão — Frente 05: mecanismo (run 20260930-0530)

## 1. Curva densa — a decisão é ainda mais cedo do que dizíamos

Union (xgboost/success): k0 **0,578** → f010 **0,735** → f025 0,758 → f050 0,782 → f100 **0,818**.

**No primeiro DÉCIMO do voo (~0,2 s, ~2 instantes de tracking) já se acumulam +0,157 = 65%
do salto total.** A curva não é um degrau no fim nem uma rampa suave: é um salto imediato
após o toque, seguido de subida gradual. Bola sozinha no f010: 0,729 — o vetor inicial
da bola (direção + qualidade do toque) é praticamente todo o sinal precoce. Campos dos
jogadores: f010 0,603 → f100 0,797 (a área resolve depois, gradual).
Robustez: no braço de tempo absoluto (frente 02), 0,4 s pós-toque = 0,745 — convenção não
muda a leitura.

## 2. Quem se mexe: atacantes ou defensores? — Redundância difusa

Canais (f100, xgboost): attack isolado 0,787 · shared 0,786 · defense 0,752 — mas o LOO de
QUALQUER canal deixa a união em 0,848-0,850 (remover até melhora marginalmente). A
informação dos jogadores está ESPALHADA e é mutuamente substituível: não há "o canal que
resolve". Estabilidade por canal: attack 0,342 / defense 0,341 — acima dos blocos de voo
(0,09-0,26), muito abaixo do strike (0,74): nível time, não individual.
Únicos de verdade (replica a frente 03): **flight3d (LOO −0,015) e arrival (−0,012)**.

## 3. Onde na área? — "ao redor da bola"

Regiões isoladas: around **0,766** · center_box 0,643 · global 0,618 · second_post 0,606 ·
first_post 0,593. LOO de qualquer região: 0,797 (= fields completo) — redundantes entre si,
mas o entorno imediato do cruzamento é disparado a mais forte isolada.

## 4. Pré-toque com a melhor família (LogReg) — objeção fechada

k−10 0,559 → k0 **0,609** (consistente com 0,604 da frente 06 na amostra pequena). Mesmo a
melhor família pré-toque conhecida trava em ~0,61 vs 0,818 do voo: **o lado esquerdo plano
da curva é real, não artefato de features**.

## Decisão: supported — o headline afia para "decidido nos primeiros instantes do voo"

A frase do paper passa de "decidido no ¼ do voo" para **"dois terços da decisão acontecem
no primeiro décimo do voo (~0,2 s)"** — pelo vetor inicial da bola. A área refina depois.
Abstract atualizado (v3). Nenhum resultado anterior muda de sinal; todos se replicam.

## Limitações

- f010 usa ≥1 instante pós-toque por construção (fração do próprio voo); o braço absoluto
  (0,4 s) confirma a leitura.
- Canais por entropia de time (ataque/defesa) não separam "corrida do recebedor" de
  "movimento coletivo" — granularidade por jogador fica para o paper completo.
- figure_curve_v2.png substitui a v1 como figura de submissão.
