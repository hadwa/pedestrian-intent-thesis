# Exp117 final report

Retrospective matched neural core56 completion; fixed threshold 0.5.
PR-AUC is average precision. Parent training and model-selection rules are unchanged.

| Case | Arm | F1 mean | PR-AUC mean | ROC-AUC mean |
|---|---|---:|---:|---:|
| mslstm_pie | G | 0.809927 | 0.853650 | 0.935398 |
| mslstm_pie | H | 0.822135 | 0.846777 | 0.934830 |
| mslstm_pie | I | 0.828921 | 0.859921 | 0.938407 |
| mslstm_pie | J | 0.822746 | 0.854871 | 0.937325 |
| mslstm_pie | K | 0.827391 | 0.862852 | 0.945671 |
| mslstm_pie | L | 0.823058 | 0.860430 | 0.936324 |
| mslstm_pie | core56 | 0.812193 | 0.851442 | 0.932063 |
| pedcmt_jaad | A | 0.673408 | 0.625618 | 0.689052 |
| pedcmt_jaad | B | 0.676727 | 0.606137 | 0.663912 |
| pedcmt_jaad | C | 0.597153 | 0.571303 | 0.606493 |
| pedcmt_jaad | D | 0.637304 | 0.660957 | 0.687890 |
| pedcmt_jaad | E | 0.653512 | 0.676195 | 0.709942 |
| pedcmt_jaad | core56 | 0.591135 | 0.632710 | 0.669930 |
| pedcmt_pie | A | 0.792614 | 0.858395 | 0.919190 |
| pedcmt_pie | B | 0.774570 | 0.821714 | 0.913965 |
| pedcmt_pie | C | 0.762484 | 0.791681 | 0.907928 |
| pedcmt_pie | D | 0.809673 | 0.789451 | 0.925997 |
| pedcmt_pie | E | 0.814212 | 0.851775 | 0.934249 |
| pedcmt_pie | core56 | 0.821133 | 0.849174 | 0.930261 |

MSLSTM references G=core, H=v1, I=full63, J=raw pose, K=GT Action/Look, L=full union.
PedCMT references A=core, B=raw pose, C=v1, D=full63, E=full union.
All reference predictions are reused, not retrained. Full per-seed metrics and sample SD are in metrics/.
No architecture or threshold was changed after test. No thesis files were edited. Stop; no further experiments.
