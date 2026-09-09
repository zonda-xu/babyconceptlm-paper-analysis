# Static figure contracts

Delivery is reproducible Python-generated PDF/PNG files in a user-selected output directory. These are user-owned publication figures, without added vendor branding. Redraws preserve the retained aggregates, not pixel-identical manuscript layout.

| Output | Analytical question and supported reading | Family / grain | Context and encodings |
| --- | --- | --- | --- |
| `exposure_curves` | How do recorded scores vary across exposure anchors? Curves describe recipe-level trajectories. | Two-panel line; 28 anchors per series, log words axis | STRICT fast macro vs multilingual zero-shot; explicit blue/gold profile roots, solid/dashed DWA distinction and markers; three language colors plus dark-neutral macro |
| `segmentation_lengths` | Where is the complete-segment length mass concentrated? Distributions describe realized length dominance. | Four-panel discrete histogram; lengths 1–8 | Same 0–85% y scale, model tokens, uncensored denominator, 2,000 sampled documents/model; single blue root |
| `fmri_<domain>` | How do retained correlations vary across within-model returned states? Descriptive, unmatched comparison only. | Two-panel bars with mean ± SE; 13 and 10 states | Six participants, unscaled correlation, shared zero-based axis, within-model indices; blue marks and dark error bars |

The layer-sweep input includes mean, SD, SE and sample count but deliberately excludes participant rows. Histogram inputs include both counts and fractions. Curve inputs retain task components as well as macros. No auxiliary raw fields are fabricated to enable other plots.

QA: verify source arithmetic, check axes and subgroup identities, render at the actual export footprint, inspect label/legend collisions and grayscale distinctions. Domain panels intentionally retain the same chart form because they ask the same within-model-state question. The scripts do not claim cross-model depth matching or paired inference from these panels.
