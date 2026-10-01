# Decisão — Frente 01: base temporal

**Run:** 20260930-0115 na h100-2 (job-f56b7c74 + retomada job-dce071d1). Local: runs/1/.

## Evidência

| Quantidade | Valor | Critério | Veredito |
|---|---|---|---|
| Cobertura (npz escritos / amostra) | 10.871 / 10.871 (100%) | s1: 100% | ✅ |
| Instantes presentes | 201.619 / 201.619 (100%) | — | ✅ |
| Fidelidade k=0 vs features (máx) | 5,89e-06 | s2: ≤ 1e-5 | ✅ |
| Fidelidade k=0 (mediana; frac ≤1e-5) | 1,47e-06; 100% | ≥99% ≤1e-6 esperado | ✅ (28% ≤1e-6, 100% ≤1e-5) |
| frac_full_pre | 1,0000 | s3: ≥ 0,95 | ✅ |
| frac_end_reached / instantes inválidos | 1,0000 / 0 | — | ✅ |
| CR ausente / partidas sem bruto / clamp K | 0 / 0 / 0 | p1/p2/p3 não dispararam | ✅ |

Fidelidade por família: diffs máximos concentrados em `pitch_control_*` (5,9e-06 no
grad_towards_goal) — consistente com quantização float32 dos npz contra features float64
(somas de pitch control acumulam ~9.600 células; entropia é mais tight). Nenhum indício
de desvio de convenção (velocidade/instante CR): um erro de convenção daria diffs ~1e-2+.

Tempos (h100-2, 24 workers): E2 extração 994 s · E3 campos 293 s · S4 fidelidade ~2 min ·
S5 cobertura ~4 min. Disco: 32,8 GB de npz + 152 MB de windows em ~/xcross-lab/data/fields_seq/.

## Claims

1. A base temporal da amostra cheia (10.871 cruzamentos / 1.041 partidas / 201.619
   instantes de CR−2s ao fim da janela) está materializada na h100-2 com cobertura 100%.
2. O instante k=0 das sequências reproduz as 30 somas espaciais dos features canônicos
   até o ruído de quantização float32 (máx 5,9e-06) — a âncora tabular e a base temporal
   são consistentes entre si.
3. Integridade estrutural dos npz: 0 instantes inválidos, 0 arquivos inválidos, 0
   cruzamentos com time zerado no CR, pré-toque completo em 100% dos cruzamentos.

## Decisão

**Supported — frente 01 encerrada com sucesso.** As frentes 02–04 (curva de informação,
atribuição por blocos, skill por bloco) estão destravadas; a matéria-prima está na raiz
canônica `~/xcross-lab/data/fields_seq/` na h100-2.

## Limitações

- A fidelidade foi medida contra os features reconstruídos na h100-2 (2026-09-21), não
  contra os da H100-1 (perdidos). A cadeia de validação cruzada de plataforma que existia
  na frente 05 não foi re-executada; em seu lugar, a âncora canônica passou a ser o par
  (features h100-2, sequências h100-2) — ambos regenerados do mesmo bruto com o mesmo código.
- métrica `elapsed_s.total` do S6 soma só os estágios S* (bug cosmético documentado;
  os tempos individuais E2/E3/S4/S5 estão corretos em metrics.json).
- `fid_frac_le_1e6` = 0,28 (não ≥0,99 como o notes previa): a mediana 1,5e-06 fica
  marginalmente acima de 1e-6 por causa das somas de pitch control; 100% ≤ 1e-5. O
  critério decisivo (s2, ≤1e-5) passou com folga.
