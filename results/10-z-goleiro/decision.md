# Decisão — Frente 10: eixo z + goleiro no voo (run 20261001-0010)

## Curvas sozinhas (xgboost/success, OOF)

| Corte | z-only | gk-only | (ref: ball2d) |
|---|---|---|---|
| k0 | 0,526 | 0,511 | 0,521 |
| f010 | 0,548 | 0,508 | **0,729** |
| f050 | 0,602 | 0,585 | 0,769 |
| f100 | 0,636 | 0,597 | **0,773** |

## Contrastes (bootstrap pareado 500)

- **z adiciona em f010: +0,0009 [−0,0035; +0,0052] — NADA.** A altura não participa do sinal precoce.
- **z adiciona em f100: +0,0120 [+0,0078; +0,0160]** — altura na chegada soma pouco e significativamente.
- z+gk na união: +0,0066 [+0,0037; +0,0092].

## Leitura: o sinal precoce é HORIZONTAL; a altura é tempero da chegada; o goleiro é figurante

1. **A decisão precoce vem da direção 2D, não da altura.** A curva do z sobe devagar
   (0,55→0,64) sem salto early; o z adiciona ~zero no primeiro décimo e só +0,012 na
   chegada — a física do "para onde" domina a do "por cima de".
2. **O goleiro é personagem menor no voo:** curva fraca (≤0,60), contribuição marginal
   na união. **Caveat estrutural:** presente em só 50,5% dos instantes (janela de
   broadcast) — o sinal dele é parcialmente invisível para nós; leitura conservadora.
3. z presente em 90,9% das linhas; imputação por mediana nos faltantes.

## Fechamento do Eixo 1 (itens 1-4 do mapa)

A física do cruzamento está completa: **direção horizontal** (decide cedo, 0,73),
**altura** (tempero da chegada, +0,012), **duelo** (a chegada em si, 0,86). Com as
frentes 06-10, os ingredientes do "O QUÊ" estão todos decompostos.

## Decisão: **supported** — z e GK medidos, papéis estabelecidos, Eixo 1 item 4 fechado
