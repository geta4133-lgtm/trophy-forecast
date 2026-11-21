# trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/geta4133-lgtm/trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/geta4133-lgtm/trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many Grand Theft Auto VI players will hold each PlayStation trophy
one week after launch: five quantiles (5/25/50/75/95%) of every trophy's
Exophase earn rate at the case's target, from the recorded history of reads.

- `main.py` — thin shell: reads every `*.in` case, writes `<id>.out` forecasts.
- `model.py` — the forecaster: per-trophy logit-scale trend fit (least-squares
  slope over `ln(1+age since release)`, damped exponentially with horizon),
  a spread from observed refresh jumps and fit residuals on top of the house's
  random-walk floor, quantiles via the standard normal's z at each level.
- Standard library only, deterministic (no clock, no randomness, no network),
  ~0.03 s per case. Python 3.13.

Run: `mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py`

License: MIT.
