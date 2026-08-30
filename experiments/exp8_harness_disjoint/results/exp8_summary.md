# Exp8 quick-read summary (auto-generated)

| feature | in_format | loho | lobo |
|---|---|---|---|
| static_full | 0.924 | 0.866 | 0.508 |
| static_pca8 | 0.786 | 0.749 | 0.444 |
| meanvel | 0.855 | 0.781 | 0.602 |
| tfidf | 0.943 | 0.919 | 0.567 |
| tags | 0.618 | 0.514 | 0.460 |
| length | 0.502 | 0.441 | 0.453 |
| tfidf+meanvel | 0.945 | 0.915 | 0.617 |

## Drops (in_format − loho mean)

- static_full: in_format 0.924 → loho 0.866 (drop 0.058)
- static_pca8: in_format 0.786 → loho 0.749 (drop 0.037)
- meanvel: in_format 0.855 → loho 0.781 (drop 0.073)
- tfidf: in_format 0.943 → loho 0.919 (drop 0.024)
- tags: in_format 0.618 → loho 0.514 (drop 0.103)
- length: in_format 0.502 → loho 0.441 (drop 0.061)
- tfidf+meanvel: in_format 0.945 → loho 0.915 (drop 0.030)

## Gates (pre-registered)

- **A_monitor_transfers: PASS**
- **B_geometry_beats_lexical_drop: FAIL**
- **C_velocity_null_recheck: PASS**
- **D_lowdim_transfers: PASS**
