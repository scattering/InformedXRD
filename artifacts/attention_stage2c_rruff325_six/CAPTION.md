# Attention Panel Notes

Six calibrated `Stage-2c` attention overlays on `RRUFF-325`.

Correct cases:
- `Arsenopyrite` (`sample_index=15`, `true EG 99`, `top-1 EG 99`)
- `Brochantite` (`sample_index=49`, `true EG 4`, `top-1 EG 4`)
- `Clinoatacamite` (`sample_index=68`, `true EG 4`, `top-1 EG 4`)

Descendant-error cases:
- `Adamite` (`sample_index=3`, `true EG 18`, `top-1 EG 4`)
- `Afwillite` (`sample_index=4`, `true EG 6`, `top-1 EG 99`)
- `Anglesite` (`sample_index=9`, `true EG 17`, `top-1 EG 4`)

Figure intent:
- Labels are restricted to well-separated peaks inside high-attention windows.
- The panel is intended as an interpretability diagnostic, not a primary quantitative result.
- The current qualitative takeaway is cautious: attention is selective and more localized than in earlier overlays, but the maps still emphasize peak-rich windows rather than cleanly isolating weak discriminatory reflections.
