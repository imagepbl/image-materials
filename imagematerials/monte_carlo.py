"""
monte_carlo.py
------------------------------------
Reads a ranges CSV (year, material, technology, min, max) and draws one
uniform random sample per (year, material, technology) pair, returning a
DataFrame in the same wide format as the original material intensities file:

    index 0 : year
    index 1 : material
    columns : technologies

Usage
-----
    from monte_carlo_material_intensities import load_ranges, sample_intensities

    ranges = load_ranges("material_intensities_ranges.csv")

    # Monte Carlo loop
    results = []
    for i in range(n_iterations):
        mi = sample_intensities(ranges)           # same shape as original CSV
        result = run_model(mi, ...)               # your model here
        results.append(result)

The sampled DataFrame can be written back to CSV with:
    mi.to_csv("sampled.csv")   # produces the same format as the original
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Sampling distributions
# ---------------------------------------------------------------------------
# Extend this dict to add other distribution types.
# Each callable receives (rng, row) and returns a scalar.

def _uniform(rng: np.random.Generator, row: pd.Series) -> float:
    lo, hi = row["min"], row["max"]
    if lo == hi:
        return lo
    return float(rng.uniform(lo, hi))


def _triangular(rng: np.random.Generator, row: pd.Series) -> float:
    """Requires columns 'min', 'mode', 'max'."""
    lo, mode, hi = row["min"], row["mode"], row["max"]
    if lo == hi:
        return lo
    return float(rng.triangular(lo, mode, hi))


DISTRIBUTIONS = {
    "uniform": _uniform,
    "triangular": _triangular,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_ranges(filepath: str) -> pd.DataFrame:
    """
    Load the ranges CSV.

    Expected columns: year, material, technology, min, max
    Optional columns: mode  (needed for triangular distribution)
                      distribution  (defaults to 'uniform')

    Returns the raw long-form DataFrame; pass it directly to sample_intensities.
    """
    df = pd.read_csv(filepath)

    required = {"year", "material", "technology", "min", "max"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ranges file is missing columns: {missing}")

    if "distribution" not in df.columns:
        df["distribution"] = "uniform"

    return df


def sample_intensities(
    ranges: pd.DataFrame,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Draw one random sample from each (year, material, technology) range.

    Parameters
    ----------
    ranges : pd.DataFrame
        Output of load_ranges().
    rng : np.random.Generator, optional
        Pass an existing Generator for reproducibility inside a loop.
    seed : int, optional
        Seed a fresh Generator (ignored when rng is supplied).

    Returns
    -------
    pd.DataFrame
        Wide-format DataFrame identical in structure to the original
        material intensities file:
            - first two index levels: year, material
            - columns: technologies
        The DataFrame is NOT indexed (year and material are plain columns),
        matching the original CSV layout.
    """
    if rng is None:
        rng = np.random.default_rng(seed)

    rows = []
    for _, row in ranges.iterrows():
        dist_name = row.get("distribution", "uniform")
        sampler = DISTRIBUTIONS.get(dist_name)
        if sampler is None:
            raise ValueError(
                f"Unknown distribution '{dist_name}'. "
                f"Available: {list(DISTRIBUTIONS)}"
            )
        value = sampler(rng, row)
        rows.append({"year": row["year"], "material": row["material"],
                     "technology": row["technology"], "value": value})

    long_df = pd.DataFrame(rows)

    # Pivot to wide format: rows = (year, material), columns = technology
    wide_df = long_df.pivot_table(
        index=["year", "material"],
        columns="technology",
        values="value",
        aggfunc="first",
    ).reset_index()

    wide_df.columns.name = None  # remove the pivot artefact name
    return wide_df


# ---------------------------------------------------------------------------
# Example / quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    ranges_file = sys.argv[1] if len(sys.argv) > 1 else "material_intensities_ranges.csv"
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    ranges = load_ranges(ranges_file)
    rng = np.random.default_rng(42)

    print(f"Running {n} Monte Carlo iterations ...\n")
    for i in range(n):
        mi = sample_intensities(ranges, rng=rng)
        print(f"--- Iteration {i + 1} ---")
        print(mi.to_string(index=False))
        print()
