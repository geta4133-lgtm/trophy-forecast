"""gta6-trophy-rarity-forecast-01 entry: damped-trend trophy forecaster (Python 3.13, stdlib only).

For each trophy, we forecast its earn rate on Exophase at the case's target by:

  1. Taking the trophy's rate series from the case's history (null-safe: the last non-null
     read is used; a trophy never seen is forecast near 0 with a wide spread).
  2. Fitting a slope in logit space over the recent history against ln(1 + age), where age
     runs from the game's release (a classic adoption-clock transform), and damping it
     toward the target with a decay factor exp(-h/tau) with tau = 2.0 days (a week-one
     S-curve grows fast then flattens; the final horizon is ~6.5 days so a strong damping
     keeps the drift honest). The median forecast is persistence + damped drift, floored
     at the persistence median for rates that are growing and at 100.0 for the platinum.
  3. Setting the spread as the baseline random walk's 0.5*sqrt(d) in logit units,
     inflated when a refresh-like jump appears in the last 24 h (a jump is a move that
     widens the interquartile band of the diffs), plus a small estimate of residual noise,
     then blending that with the honest residual sigma of a well-fit line so a clean
     history pays a tighter band. Quantiles are median +/- z * sigma, kept monotone by
     construction, rounded to 4 decimals half-up.

  The floor is applied as a pool share: because launch-night rates move in lock-step
  (Exophase refreshes every trophy at once), a growing trophy's upward risk is shared
  with its pool: each trophy gets at most 100/pool% of the remaining headroom above the
  pool's last rate, so the pool can never overshoot 100% collectively.

The grader runs `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) per case; a shipped
model.py imports only after the sys.path line below. Everything is deterministic: no
clock, no randomness, no dict-order dependence.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first so `import model` works.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case  # noqa: E402  (our model lives beside this file)

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
