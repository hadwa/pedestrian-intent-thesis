# Approved complete-input cohort amendment

Approved by the user on 2026-09-12 after the original full-cohort speed
preflight blocked. `BLOCKED.json`, `SOURCE_PREFLIGHT.md`, and the original frozen
`DESIGN.md` are retained unchanged as history.

The amended experiment excludes every training window from exactly
`video_0032`, `video_0055`, and `video_0093`, the three PSI2 TrainVal videos
without official CV/speed annotation members. This removes 143, 141, and 301
windows respectively (585 total), leaving 26,295 training windows. Validation
has none of these videos and remains unchanged at 14,922 windows. Both S187 and
R247 use these identical rows in identical order. Results are an amended
complete-input-cohort experiment and must never be described as using the same
training population as Exp113.

All other frozen rules remain unchanged: native next-frame16 labels; 15 observed
frames; FIRST7/LAST8 for all temporal and road summaries; exact official numeric
`speed(km/hr)` with no missing-value imputation; historical deduplicated 161-D
BBox/BIU union plus 21 temporal and five ego features; exact Exp075 TwinLiteNet+
Large checkpoint/adapter and 60-D road representation; seeds42/43/44; six total
fits; fixed predictor threshold .5; train/validation only; no test access.

Before extraction, preparation must prove complete finite speed coverage for
every observed frame of all retained windows and exact cohort equality between
both arms. Any additional missing video/frame, malformed speed, population
change, or source drift stops the run.
