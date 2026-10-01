# Decisão — Frente 09: intenção × execução (run 20260930-2320)

## Decomposição do sinal precoce da bola (xgboost/success, OOF)

| Modelo | AUC | O que é |
|---|---|---|
| **A destino** (end_x/y + geometria) | **0,737** | para onde a bola foi |
| B execução f010 (trajetória no 1º décimo) | 0,729 | como saiu do pé |
| B execução f020 | 0,746 | idem, 2º décimo |
| **C destino + execução** | **0,777** | os dois juntos |
| D execução pura (resíduos ⊥ destino) | 0,730 | batida além do alvo |

## Contrastes (bootstrap pareado 500)

- **C − A = +0,040 [+0,034; +0,045]** — a execução soma sobre o destino, com IC folgado.
- **A − B = +0,009 [−0,001; +0,018]** (f010) e −0,009 [−0,016; +0,001] (f020) — destino e
  trajetória precoce são estatisticamente indistinguíveis um do outro.
- **D − A ≈ −0,006 [IC cruza 0]** — execução pura ≈ destino em força.

## Leitura: COMPLEMENTARIDADE, não dominação

Minha hipótese forte ("o sinal precoce é a escolha do alvo; selection over execution
dentro da entrega") é **refutada na forma forte e confirmada na fraca**:

1. **Destino e batida carregam sinais de força equivalente (~0,73 cada)** e
   **complementares** (juntos 0,777): nem "só importa onde", nem "só importa como".
2. Cada um sozinho ≈ o outro; nenhuma das duas decomposições domina (ICs cruzam zero).
3. **Consistente com o crédito por papel (frentes 03/04):** a qualidade da entrega
   (alvo + batida) resolve o desfecho cedo, mas o bloco de técnica NÃO é skill
   individual estável (0,09-0,12) — é propriedade do EVENTO entrega (situação + execução
   daquele cruzamento), não do cruzador como traço.

**A frase fina:** a bola precoce é destino E batida, em partes iguais e
complementares — e ambas são obras do momento, não do jogador.

## Critérios

s1 destino forte (≥0,65) ✓ · s2 alvo ≈ trajetória (|Δ|≤0,02) ✓ · s3 execução soma
(IC>0) ✓ — os três passaram, com a leitura honesta de complementaridade.

## Limitações

- Destino é quantidade a posteriori (descritivo, lado direito da curva); a versão
  "alvo extrapolável em tempo real" fica para o paper completo.
- D por OLS global (sem rótulo — não vaza outcome); residualização não-linear
  (GBM residuals) pode superestimar levemente D.
- Só corte xy (npz sem z); a decomposição com z (apex/ângulo) enriquece no paper completo.

## Decisão: **supported — complementaridade destino×batida; hipótese forte refutada**
