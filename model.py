"""The forecasting model: damped trend + honest spread, in logit space.

Split from main.py so the entrypoint stays a thin IO shell (like the kit's starter).
Pure stdlib, fully deterministic: no clock, no randomness, no set/dict iteration order
that could change between runs.
"""

import math

# ---------------------------------------------------------------- constants ----

# The quantile levels the contest scores, in the order each output row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (fixed by the contest).
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

# The house baseline's random-walk spread in logit units per square-root day. We reuse
# it as our floor so a quiet trophy is never narrower than the baseline.
SIGMA_WALK = 0.5
# Damping time constant (days) for the trend: exp(-h/tau). Week-one adoption curves
# flatten on a ~1-2 day scale; 2.0 keeps a modest, conservative drift at long horizons.
TAU = 2.0
# How many trailing reads the trend fit uses (30-minute clock: 16 reads = last 8 h).
FIT_WINDOW = 16
# Age transform for the trend fit: growth is fastest right after release and slows,
# so logit rate vs ln(1 + age) is close to linear over the fit window.
RELEASE_ISO = "2026-11-19T00:00:00Z"


# ------------------------------------------------------------------ helpers ----

def clamp(x: float) -> float:
    return min(99.99, max(0.01, x))


def logit(pct: float) -> float:
    return math.log(clamp(pct) / (100 - clamp(pct)))


def expit(lg: float) -> float:
    return 100.0 / (1.0 + math.exp(-lg))


def round4(x: float) -> float | int:
    """x rounded to 4 decimals, half up, as JavaScript's Math.round(x*10000)/10000 does.

    A whole number comes back as an int so json.dumps writes 100, not 100.0.
    """
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_iso(s: str) -> float:
    """ISO 8601 UTC timestamp -> seconds since epoch (deterministic, stdlib-free parsing).

    Only the fixed shape 'YYYY-MM-DDTHH:MM:SSZ' (or with .mmm) the house uses is needed.
    """
    d, t = s[:10], s[11:19]
    days = _days_from_civil(int(d[0:4]), int(d[5:7]), int(d[8:10]))
    secs = days * 86400 + int(t[0:2]) * 3600 + int(t[3:5]) * 60 + int(t[6:8])
    return secs


def _days_from_civil(y: int, m: int, d: int) -> int:
    """Days since 1970-01-01 (Howard Hinnant's algorithm, no datetime needed)."""
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def _least_squares(xs: list[float], ys: list[float]) -> tuple[float, float]:
    """Simple least squares slope/intercept; returns (0, mean(ys)) for degenerate input."""
    n = len(xs)
    if n < 2:
        return 0.0, (sum(ys) / len(ys)) if ys else 0.0
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return 0.0, my
    sxy = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    return sxy / sxx, my - (sxy / sxx) * mx


def _median(a: list[float]) -> float:
    b = sorted(a)
    n = len(b)
    return b[n // 2] if n % 2 else 0.5 * (b[n // 2 - 1] + b[n // 2])


def _quartile_spread(a: list[float]) -> float:
    """IQR/1.349 of the diffs: a robust sigma estimate that a single refresh jump
    barely moves (the jump lands in one tail, the quartiles stay put)."""
    if len(a) < 4:
        return 0.0
    b = sorted(a)
    n = len(b)
    q1 = b[n // 4]
    q3 = b[(3 * n) // 4]
    return max(0.0, (q3 - q1) / 1.349)


# ------------------------------------------------------------------ forecast ----

def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    if not case["history"]:
        raise ValueError(f"{name} has no history")

    horizon_d = case["horizon_minutes"] / 1440.0
    history = case["history"]
    n_trophies = len(case["trophies"])
    reads_at = [parse_iso(r["at"]) for r in history]

    # --- collect the logit series of every trophy and how long it stayed null ---
    series: list[list[float | None]] = []
    first_seen: list[int] = []
    for i in range(n_trophies):
        col: list[float | None] = []
        first = -1
        for j, r in enumerate(history):
            v = r["rates"][i]
            col.append(logit(v) if v is not None else None)
            if v is not None and first < 0:
                first = j
        series.append(col)
        first_seen.append(first)

    rows = []
    for i, trophy in enumerate(case["trophies"]):
        lg_last = series[i][-1]
        # Never-seen trophy: forecast ~0 with a wide band (the baseline uses logit(0),
        # i.e. the 0.01% clamp; we keep that point but widen nothing extra - the walk
        # spread already gives a wide band).
        lg_last = -6.9 if lg_last is None else lg_last

        # A trophy still unseen in a week-one history is likely a late-game spoiler:
        # keep a wide band rather than a tight drift (handled via jump inflation below).

        # --- trend fit in logit space against ln(1 + age) over the fit window ---
        age0 = parse_iso(RELEASE_ISO)
        xs: list[float] = []
        ys: list[float] = []
        start = max(0, len(history) - FIT_WINDOW)
        for j in range(start, len(history)):
            v = series[i][j]
            if v is not None:
                xs.append(math.log1p((reads_at[j] - age0) / 86400.0))
                ys.append(v)
        slope, intercept = _least_squares(xs, ys)
        x_target = math.log1p((parse_iso(case["target"]) - age0) / 86400.0)

        # Damped drift toward the target; fitted-at-window-start so the slope is the
        # recent growth, not extrapolated from today.
        x_fit0 = xs[0] if xs else 0.0
        drift = slope * (x_target - x_fit0) * math.exp(-horizon_d / TAU)

        # Growing trophies never forecast below persistence: in week one a rate's
        # true drift is essentially always up, and the refresh steps levels down once.
        # (A trophy at 100% has no headroom by construction: logit clamps it, its
        # slope is 0, and its upper quantiles ride the clamp toward 100.)
        lg_med = lg_last + max(0.0, drift)

        # --- spread: house walk floor + jump inflation + residual blend ---
        spread = SIGMA_WALK * math.sqrt(horizon_d)
        if len(history) >= 8:
            # Diffs of the logit series over the whole history (whole-history diffs are
            # what a refresh shows: one big step among many zeros).
            vals = [v for v in series[i] if v is not None]
            if len(vals) >= 4:
                diffs = [vals[j + 1] - vals[j] for j in range(len(vals) - 1)]
                # Absolute jump size of the biggest recent move, per square-root day.
                jump = max(abs(d) for d in diffs)
                sigma_jump = (jump / 2.0) * math.sqrt(horizon_d) if jump > 0.2 else 0.0
                # Residual sigma of the line fit: honest noise if the history is clean.
                fitted = [intercept + slope * x for x in xs]
                resid = [ys[j] - fitted[j] for j in range(len(xs))]
                sigma_resid = _quartile_spread(resid) * math.sqrt(max(horizon_d, 1e-6))
                spread = max(spread, sigma_jump, sigma_resid)
            # A trophy that only just appeared has genuine uncertainty (is it an early
            # spoiler or a late-game one?): widen a little.
            if 0 <= first_seen[i] and (len(history) - 1 - first_seen[i]) <= 3:
                spread += 0.25 * math.sqrt(horizon_d)

        q = [expit(lg_med + z * spread) for z in Z]
        rows.append({"id": trophy["id"], "q": [round4(x) for x in q]})

    return {"trophies": rows}
