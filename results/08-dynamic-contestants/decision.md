# Decisão — Frente 08: recebedor dinâmico (run 20260930-0900)

## Os três resultados (critérios s1/s2/s3 todos True)

### 1. O vazamento foi confirmado e eliminado (s1)

| Corte | sep contaminada (f07) | **sep dinâmica (limpa)** |
|---|---|---|
| k−10 | 0,694 | **0,515** |
| k0 | 0,747 | **0,518** |
| f100 | 0,879 | 0,857 |

O pré-toque da sep colapsa para o nível de ruído (~0,52) quando a identidade dos
rastreados é definida sem informação futura — o 0,69–0,75 da frente 07 era
exatamente o vazamento por seleção de identidade que declaramos. **A prática de
declarar funcionou: diagnóstico confirmado, versão limpa construída.**

### 2. O recebedor EMERGE tarde — achado novo

P(dinâmico = recebedor da chegada): **5% em f=0,1 · 18% em f=0,25 · 28% em f=0,5 ·
100% na chegada.** Média de 3,1 trocas de identidade por cruzamento. Durante quase
todo o voo, o "atacante mais próximo da bola" NÃO é quem vai disputá-la — o elenco do
duelo só cristaliza no final.

### 3. A separação dinâmica diverge TARDE (figura corrigida)

| fração do voo | sucesso | fracasso | gap |
|---|---|---|---|
| 0,1 | 9,13 m | 9,42 m | **0,29 m (nada)** |
| 0,5 | 7,00 m | 7,78 m | 0,78 m |
| 0,7 | 4,08 m | 5,34 m | **1,26 m** |
| 1,0 | 3,02 m | 5,05 m | **2,03 m** |

O "gap de 2 m desde o início" da frente 07 era o artefato da identidade. A dinâmica
verdadeira: o contest é indiferenciado no começo do voo (~9 m entre os mais próximos,
sem diferença por desfecho) e **separa na segunda metade do voo**, convergindo com a
cristalização da identidade.

## O mecanismo em duas fases (síntese final, reconciliando tudo)

1. **Fase 1 — o destino (10–30% do voo):** a TRAJETÓRIA DA BOLA prediz o desfecho
   (0,73 no primeiro décimo; frente 05), mas o contest humano ainda é indiferenciado
   (gap ~0) e o recebedor é irreconhecível (5–18%).
2. **Fase 2 — a resolução (últimos 30–40%):** o duelo cristaliza — identidade converge,
   separação diverge por desfecho (0,8 → 2,0 m) e a chegada vira o sinal mais forte
   (0,857; união+dyn 0,887).

**A bola é o destino; os humanos são a resolução.** A previsibilidade precoce vem de
onde a bola vai; a informação da chegada é de quem a disputa.

## Decisão: **supported** — limitação da frente 07 resolvida, mecanismo refinado

A sep dinâmica substitui a contaminada em qualquer uso futuro. O achado da emergência
tardia do recebedor (P convergindo 5%→100%) é novo na literatura e vira figura do
paper completo.

## Limitações

- "Mais próximo da bola" é uma definição de proximidade; emergência pode ser testada
  com outras definições (mais próximo do ponto de queda previsto,etc.).
- dyn f100 ≈ arrival por construção (na chegada, dinâmico = chegada) — a comparação
  honesta é nos cortes intermediários.
- Sem ICs bootstrap nesta frente (pontos claros; paper completo adiciona).
