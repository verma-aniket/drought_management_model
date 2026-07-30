import numpy as np
import pandas as pd
from scipy.stats import gamma, norm

def compute_spi(precip, k, return_params=False):
    """
    Compute SPI-k (k-month Standardized Precipitation Index) with H-correction.

    precip: pandas Series with datetime index at monthly frequency
    k: aggregation window (e.g., 3, 6, 12)
    return_params: if True, return the fitted gamma distribution parameters for each month
    """

    if return_params:
        # create empty dictionary to store parameters for each month
        params = {}

    # --- 1. Compute k-month aggregated precipitation ---
    Pk = precip.rolling(k).sum()

    SPI = pd.Series(index=Pk.index, dtype=float)

    # --- 2. Fit gamma distribution per calendar month ---
    for month in range(1, 13):
        # Extract all historical Pk values for this calendar month
        vals = Pk[Pk.index.month == month].dropna()

        if len(vals) < 5:
            # Not enough samples to reliably compute SPI
            continue

        # --- (a) Compute probability of zero precipitation (q) ---
        zeros = np.sum(vals == 0)
        q = zeros / len(vals)

        # Gamma cannot take zero → adjust only nonzero values
        nonzero_vals = vals[vals > 0]

        # --- (b) Fit gamma only to non-zero values ---
        if len(nonzero_vals) > 1:
            a, loc, scale = gamma.fit(nonzero_vals, floc=0)
        else:
            # If all-zero or nearly all-zero, SPI not meaningful
            continue

        # --- (c) Compute CDF for all observations this month ---
        Pk_m = Pk[Pk.index.month == month]

        # Compute G(x) only for nonzero, set G(0)=0
        G = np.zeros(len(Pk_m))
        nz_idx = Pk_m.values > 0
        G[nz_idx] = gamma.cdf(Pk_m.values[nz_idx], a, loc=0, scale=scale)

        # --- (d) H correction ---
        # H = q + (1-q)*G
        H = q + (1 - q) * G

        # Bound H away from 0 and 1 to avoid ±inf
        H = np.clip(H, 1e-10, 1 - 1e-10)

        # --- (e) Convert to SPI by standard normal quantile ---
        SPI.loc[Pk_m.index] = norm.ppf(H)

        # if requested, store parameters
        if return_params:
            params[month] = (a, scale, q)

    if return_params:
        return SPI, params
    else:
        return SPI
    
def compute_annual_cf(
    hist_df,
    future_df,
    hist_year=2020,
    target_years=(2030, 2040, 2050),
    window=20,
    month_col="Month",
    gcm_col="GCM",
    var_col="Variable",
    exp_col="Experiment",
    value_col="Value",
):
    """
    Compute annual CFs for each Month, GCM, Variable, Experiment using:
      - CF = future_value / historic_value
      - future_value = mean over window centered on each target_year
      - historic CF = 1 at hist_year
      - linear interpolation for all years between hist_year and last target_year
    """

    # --- 1) Expand hist_df to all Experiments ---
    exps = future_df[exp_col].unique()
    hist_expanded = (
        hist_df.assign(key=1)
        .merge(pd.DataFrame({exp_col: exps, "key": 1}), on="key")
        .drop(columns="key")
        .rename(columns={value_col: "HistValue"})
    )
    hist_expanded["Year"] = hist_year

    # --- 2) Build windowed averages for each target year ---
    future_avg_list = []
    if type(target_years) != type(int(1)):
        for t in target_years:
            start = t - window // 2
            end   = start + window - 1
            mask = (future_df["Year"] >= start) & (future_df["Year"] <= end)

            fa = (
                future_df.loc[mask]
                .groupby([month_col, gcm_col, var_col, exp_col], as_index=False)[value_col]
                .mean()
                .rename(columns={value_col: "FutureValue"})
            )
            fa["Year"] = t
            future_avg_list.append(fa)
    else:
        t = target_years
        start = t - window // 2
        end = start + window - 1
        mask = (future_df["Year"] >= start) & (future_df["Year"] <= end)

        fa = (future_df.loc[mask].groupby([month_col, gcm_col, var_col, exp_col], as_index=False)[
            value_col].mean().rename(columns={value_col: "FutureValue"}))
        fa["Year"] = t
        future_avg_list.append(fa)


    future_avg = pd.concat(future_avg_list, ignore_index=True)

    # --- 3) Merge historic values onto future averages ---
    merged_fut = future_avg.merge(
        hist_expanded[[month_col, gcm_col, var_col, exp_col, "HistValue"]],
        on=[month_col, gcm_col, var_col, exp_col],
        how="left"
    )

    # Historic row (CF=1)
    hist_rows = hist_expanded.copy()
    hist_rows["FutureValue"] = hist_rows["HistValue"]
    hist_rows["CF"] = 1.0
    hist_rows.loc[hist_rows['Variable'] != 'pr', 'CF'] = 0.0

    # --- 4) Compute CFs at target years ---
    merged_fut['CF'] = 0.0
    merged_fut.loc[merged_fut['Variable'] == 'pr', 'CF'] = (
            merged_fut.loc[merged_fut['Variable'] == 'pr', 'FutureValue'] /
            merged_fut.loc[merged_fut['Variable'] == 'pr', 'HistValue'])
    merged_fut.loc[merged_fut['Variable'] != 'pr', 'CF'] = (
            merged_fut.loc[merged_fut['Variable'] != 'pr', 'FutureValue'] -
            merged_fut.loc[merged_fut['Variable'] != 'pr', 'HistValue'])

    # Combine all anchor CFs
    anchors = pd.concat([
        merged_fut[[month_col, gcm_col, var_col, exp_col, "Year", "CF"]],
        hist_rows[[month_col, gcm_col, var_col, exp_col, "Year", "CF"]]
    ], ignore_index=True)

    # --- 5) Create ANNUAL grid ---
    min_year = hist_year
    if type(target_years) != type(int(1)):
        max_year = max(target_years)
    else:
        max_year = target_years
    annual_years = pd.DataFrame({"Year": np.arange(min_year, max_year + 1)})

    base = (
        anchors[[month_col, gcm_col, var_col, exp_col]]
        .drop_duplicates()
        .assign(key=1)
        .merge(annual_years.assign(key=1), on="key")
        .drop(columns="key")
    )

    # --- 6) Merge anchor CFs and interpolate annually ---
    out = base.merge(
        anchors,
        on=[month_col, gcm_col, var_col, exp_col, "Year"],
        how="left"
    ).sort_values([gcm_col, exp_col, var_col, month_col, "Year"])

    out["CF"] = (
        out.groupby([gcm_col, exp_col, var_col, month_col])["CF"]
        .apply(lambda s: s.interpolate(method="linear", limit_direction="both"))
        .reset_index(level=[0,1,2,3], drop=True)
    )

    return out[["Year", month_col, gcm_col, exp_col, var_col, "CF"]]

def rolling_sum_nan(x, k):
    """
    Right-aligned rolling sum with first k-1 values as NaN.
    Equivalent to pandas rolling(k).sum()
    """
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)

    cumsum = np.cumsum(x)
    out[k-1:] = cumsum[k-1:] - np.concatenate(([0], cumsum[:-k]))

    return out


def calculate_spi(precip, spi_params, k=12):
    """
    Calculate Standardized Precipitation Index (SPI)

    Assumptions:
    - precip[0] corresponds to January
    - Monthly time series
    - len(precip) is divisible by 12
    - spi_params indexed 0=Jan, ..., 11=Dec
    """

    # --- 1. Rolling sum (preserve structure) ---
    Pk = rolling_sum_nan(precip, k)

    # --- 2. Reshape into (years × months) ---
    n_years = len(Pk) // 12
    Pk_2d = Pk.reshape(n_years, 12)

    spi_2d = np.full_like(Pk_2d, np.nan)

    # --- 3. Month-wise SPI ---
    for month in range(12):

        a, scale, q = spi_params[month]

        x = Pk_2d[:, month]

        valid = ~np.isnan(x)
        nz = (x > 0) & valid

        G = np.zeros_like(x)
        G[nz] = gamma.cdf(x[nz], a, loc=0, scale=scale)

        H = q + (1 - q) * G
        H = np.clip(H, 1e-10, 1 - 1e-10)

        spi_vals = np.full_like(x, np.nan)
        spi_vals[valid] = norm.ppf(H[valid])

        spi_2d[:, month] = spi_vals

    # --- 4. Flatten back to 1D ---
    spi = spi_2d.reshape(-1)

    return spi[k:] # return SPI values starting from month k


def get_spi_month(Pk, params):
    """
    Calculate Standardized Precipitation Index (SPI) given a 
    single precipitation value.

    Assumptions:
    - Pk is the rolling sum precip of k months
    - params is list or tuple of the form [a, scale, q], based on aggregation period k
    and indexed at month corresponding to Pk value
    """
    a, scale, q = params
    G = gamma.cdf(Pk, a, loc=0, scale=scale)
    H = q + (1 - q) * G
    H = np.clip(H, 1e-10, 1 - 1e-10)
    spi = norm.ppf(H)

    return spi

# define function to identify drought events
def identify_drought_events(spi, k1, k2, m):
    """
    Drought event definition: 
       - occurs any time the SPI is continuously below k1 AND
       - reaches an intensity of k2 or less FOR AT LEAST 
       - m months between start and end of event

    Extract drought events from SPI series based on run theory.

    spi: array of spi values
    k1: SPI clipping threshold (e.g., 0)
    k2: drought intensity threshold (e.g., -1)
    m: minimum duration of drought intensity threshold being met in months
    """

    below = spi < k1  # boolean array
    # find indices where runs start and end using diff
    # pad with False to capture edges
    d = np.diff(below.astype(int))
    starts = np.where(d == 1)[0] + 1
    ends = np.where(d == -1)[0] + 1

    # handle if starts/ends at boundaries
    if below[0]:
        starts = np.r_[0, starts]
    if below[-1]:
        ends = np.r_[ends, len(below)]

    drought = np.zeros(len(spi), dtype=int)
    # now iterate over runs (usually few) — this loop only goes over runs, not households
    for s, e in zip(starts, ends):
        # count how many months in this run have spi <= k2
        k2_count = np.sum(spi[s:e] <= k2)
        if k2_count >= m:
            drought[s:e] = 1

    return drought

def get_num_time_drought(drought_status):
    """
    function that returns the number of droughts and the index when they start
    given an array of drought status (0 - no drought, 1 - drought)     
    """

    # quick check to bypass function if there are no droughts
    if np.sum(drought_status) == 0:
        return 0, -1
    else:
        start_in_drought = False                                # boolean to track if initially in drought
        status_diff = np.diff(drought_status)                   # compute difference in drought status
        status_switch = status_diff[status_diff != 0]           # extract all flips (entering or exiting drought)
        if status_switch[0] == -1:                              # check if first flip is exiting drought
            start_in_drought = True                             # this means starting staus was in drought
            status_switch[0] = 1                                # update switch vector
        num_drought = np.sum(status_switch[status_switch > 0])  # compute number of droughts
        time_drought = np.where(status_diff == 1)[0] + 1        # determine index of start of drought
        if start_in_drought:                                    # special case if started in drought
            time_drought = np.concatenate((np.array([0]), time_drought))
        
        return num_drought, time_drought 

# greedy solution that extracts smallest set of scenarios where drought occurs at least once each year
def minimal_year_cover(years_list, penalty_power=1):
    """
    Greedy set cover that prefers smaller sub-lists.

    penalty_power controls how strongly to prefer small lists.
    Higher values → stronger bias.
    """

    valid = [(i, set(lst)) for i, lst in enumerate(years_list) if lst]

    remaining = set().union(*(s for _, s in valid))
    selected = []

    while remaining:

        def score(item):
            i, s = item
            new_cover = len(s & remaining)
            if new_cover == 0:
                return 0

            # penalize large sets
            return new_cover / (len(s) ** penalty_power)

        best_idx, best_set = max(valid, key=score)

        selected.append(best_idx)
        remaining -= best_set

        valid = [(i, s) for i, s in valid if i != best_idx]

    return selected

def minimal_year_cover_random(years_list, penalty_power=1, rng=None):
    """
    Greedy set cover that prefers smaller sub-lists.

    penalty_power controls how strongly to prefer small lists.
    Higher values → stronger bias.

    Randomzied to avoid bias toward early indices
    """

    valid = [(i, set(lst)) for i, lst in enumerate(years_list) if lst]

    remaining = set().union(*(s for _, s in valid))
    selected = []

    while remaining:

        scores = []
        for item in valid:
            i, s = item
            new_cover = len(s & remaining)

            if new_cover == 0:
                scores.append(0)
            else:
                score = new_cover / (len(s) ** penalty_power)
                scores.append(score)

        scores = np.array(scores)

        # find top candidates
        best_score = scores.max()
        candidates = np.where(scores == best_score)[0]

        # random selection among best
        chosen_idx = rng.choice(candidates)

        best_i, best_set = valid[chosen_idx]
        selected.append(best_i)

        remaining -= best_set
        valid.pop(chosen_idx)

    return selected

def cc_df_to_mat(df, gcms, scens, ids, var, num_months=312):
    """
    Transforms long-form climate DataFrames into flat dimensional matrices.
    """
    cc_mat = np.zeros([len(gcms) * len(scens) * len(ids), num_months])
    i = 0
    for gcm in gcms:
        for scen in scens:
            for id in ids:
                mask = (df['GCM'] == gcm) & (df['Experiment'] == scen) & (df['Set'] == id)
                cc_mat[i] = df.loc[mask, var].to_numpy()
                i += 1
    return cc_mat