# reads raw WeaGETS output and reformats it to pass into Santa Cruz rainfall-runoff model
import sys
from pathlib import Path
import numpy as np

# 1. Dynamically locate the absolute root directory of this repository.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent # climb three levels to get to the main repo folder

# Append repo root folder to the Python system path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# read config to access file strucutre
from src.utils import config

# define mean daily percentage (p) of annual daytime hours for 37.5 deg latitude (Santa Cruz)
# took average of 35 and 40 deg value from this source: https://www.fao.org/4/s2022e/s2022e07.htm#3.1.3%20blaney%20criddle%20method
# based on what the rainfall runoff modeled used to compute daily reference evapotranspiration 
pr = {1: 0.225, 2: 0.245, 3: 0.27, 4: 0.295, 5: 0.315, 6: 0.33, 
      7: 0.325, 8: 0.305, 9: 0.28, 10: 0.25, 11: 0.225, 12: 0.215}

def get_leap_year_indices(start_year: int, z_years: int) -> list[int]:
    """
    Returns 0-based indices of leap years in a block of `z_years` starting at `start_year`.
    
    Handles standard calendar rules (including century non-leap years like 2100).
    """
    leap_indices = []
    
    for i in range(z_years):
        current_year = start_year + i
        
        # Standard leap year check:
        # Divisible by 4 AND (not divisible by 100 OR divisible by 400)
        is_leap = (current_year % 4 == 0) and (
            current_year % 100 != 0 or current_year % 400 == 0
        )
        
        if is_leap:
            leap_indices.append(i)
            
    return leap_indices

def impute_leap_days(mat: np.ndarray, leap_year_indices: list[int] = None) -> np.ndarray:
    """
    Imputes Feb 29 weather data using a 1-NN approach across chunked sets.

    Parameters
    ----------
    mat : np.ndarray
        Shape (y, z, 365, 3) where:
        y = number of sets
        z = number of years per set
        365 = days per year (0-indexed: Feb 28 = index 58, Mar 1 = index 59)
        3 = variables [Tmax, Tmin, Precip]
    leap_year_indices : list of int, optional
        Zero-based year indices within each set `z` that are leap years (e.g., [3, 7, 11...]).
        If None, defaults to standard 4-year cycle (index 3, 7, 11...).

    Returns
    -------
    imputed_mat : np.ndarray
        Shape (y, z, 366, vars_count) with imputed Feb 29 inserted between Feb 28 and Mar 1.
    """
    y, z, days, vars_count = mat.shape
    assert days == 365, "Input matrix must have 365 days per year."

    # Day indices in non-leap 365-day calendar
    FEB28_IDX = 58
    MAR01_IDX = 59

    # Default leap year pattern: years 3, 7, 11... (every 4th year)
    if leap_year_indices is None:
        leap_year_indices = [yr for yr in range(z) if (yr + 1) % 4 == 0]

    # Pre-allocate output array for 366 days
    imputed_mat = np.zeros((y, z, 366, vars_count), dtype=mat.dtype)

    for set_idx in range(y):
        set_data = mat[set_idx] 

        for yr_idx in range(z):
            # Copy original 365 days up to Feb 28 (0 to 58 inclusive -> 59 days)
            imputed_mat[set_idx, yr_idx, :59] = set_data[yr_idx, :59]
            # Copy Mar 1 to Dec 31 into shifted positions (59 to 364 original -> 60 to 365 output)
            imputed_mat[set_idx, yr_idx, 60:] = set_data[yr_idx, 59:]

            # If it's a leap year, perform 1-NN imputation for index 59 (Feb 29)
            if yr_idx in leap_year_indices:
                # 1. Target vector for this leap year: average of its own Feb 28 and Mar 1 (Tmax, Tmin only)
                target_temp = np.mean(
                    [set_data[yr_idx, FEB28_IDX, :2], set_data[yr_idx, MAR01_IDX, :2]], 
                    axis=0
                )

                # 2. Build candidate pool from ALL OTHER years within this set
                candidate_days = []
                for donor_yr in range(z):
                    if donor_yr == yr_idx:
                        continue  # Exclude self
                    # Add Feb 28 and Mar 1 from donor year
                    candidate_days.append(set_data[donor_yr, FEB28_IDX])
                    candidate_days.append(set_data[donor_yr, MAR01_IDX])
                
                candidate_days = np.array(candidate_days)  # Shape: (2*(z-1), 3)

                # 3. Calculate Euclidean distance on [Tmax, Tmin] (first 2 columns)
                candidate_temps = candidate_days[:, :2]
                distances = np.linalg.norm(candidate_temps - target_temp, axis=1)

                # 4. Pick nearest neighbor (1-NN)
                best_match_idx = np.argmin(distances)
                best_match_day = candidate_days[best_match_idx]  # Full array: [Tmax, Tmin, Precip]

                # 5. Assign match to Feb 29 (index 59)
                imputed_mat[set_idx, yr_idx, 59] = best_match_day
            else:
                # Non-leap years: duplicate Feb 28 into Feb 29 position or leave as nan / placeholder
                # Here we fill with np.nan to explicitly mark non-leap Feb 29s
                imputed_mat[set_idx, yr_idx, 59] = np.nan

    return imputed_mat

# define simulation parameters
num_year = 100
start_year = 2029

# get files to loop over
files = [path for path in config.RAW_WEATHER_DIR.iterdir() if path.is_file()]

# read data into numpy array
print(files[0])
mat = np.loadtxt(files[0], delimiter=",")
num_days = mat.shape[0]

# add reference evaporation in mm/day to the matrix data using the Blaney-Criddle equation
days_in_months = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31] # Define exact days per month for a standard 365-day year
month_vector = np.repeat(np.arange(1, 13), days_in_months)  # Build 1D array of length 365: [1, 1... 1, 2, 2... 2, ..., 12, 12... 12, Shape: (365,)
pr_vector = np.array([pr[m] for m in month_vector])
pr_column = np.tile(pr_vector, num_days // 365)
t_max = mat[:, 1]
t_min = mat[:, 2]
t_mean = (t_min + t_max) / 2.0
et_column = (pr_column * (0.46 * t_mean + 8.0))[:, np.newaxis]
mat = np.hstack([mat, et_column])

# reshape data
num_vars = mat.shape[1]
tot_years = num_days // 365
num_sets = tot_years // (num_year + 1) # + 1 because we will remove 1 year from each set
mat = mat.reshape(num_sets, num_year + 1, 365, num_vars)

# build output file

# impute leap days
leap_year_idx = get_leap_year_indices(start_year, num_year+1)
leap_mat = impute_leap_days(mat, leap_year_idx)

# Days to trim from the start of Year 1 (Jan 1 to Sep 30 = 273 days in non-leap, 274 in leap)
start_is_leap = (start_year % 4 == 0 and start_year % 100 != 0) or (start_year % 400 == 0)
start_trim = 274 if start_is_leap else 273

# Days to trim from the end of the final year (Oct 1 to Dec 31 = 92 days in both leap and non-leap)
# (October has 31 days, November 30, December 31 -> always 92 days total)
end_trim = 92

# define constants for daily to monthly matrix manipulation 

# Days per month lookup
DAYS_NON_LEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_LEAP     = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Month slicing boundaries for 366-day padded year
month_slice_bounds = np.cumsum([0] + DAYS_LEAP)

# Define time vectors across the set length (e.g., 101 years)
years = np.repeat(np.arange(start_year, start_year + num_year + 1), 12)
months = np.tile(np.arange(1, 13), num_year + 1)

# Generate 'Days' vector accounting for leap years
days_list = []
for yr in range(start_year, start_year + num_year + 1):
    is_leap = (yr % 4 == 0) and (yr % 100 != 0 or yr % 400 == 0)
    days_list.extend(DAYS_LEAP if is_leap else DAYS_NON_LEAP)
days_vec = np.array(days_list)

# Generate Water Year vector: WY = Year + 1 for Oct-Dec (Months 10, 11, 12)
water_years = np.where(months >= 10, years + 1, years)

# save files
for n in range(num_sets):
    if n >0:
        break

    # ---
    # Build the monthly dataset for the rainfall runoff model
    # leap_mat[n] has shape (z, 366, 4) -> [Precip, Tmax, Tmin, ET]
    z_years = leap_mat[n].shape[0]
    monthly_data = []

    for yr in range(z_years):
        year_data = leap_mat[n, yr]  # Shape: (366, 4)
        
        for m in range(12):
            start_day = month_slice_bounds[m]
            end_day = month_slice_bounds[m + 1]
            month_data = year_data[start_day:end_day]  # Slice for month m

            # Aggregations using NaN-safe functions
            precip_sum = np.nansum(month_data[:, 0])    # Column 0: Sum Precip
            tmax_avg   = np.nanmean(month_data[:, 1])   # Column 1: Mean Tmax
            tmin_avg   = np.nanmean(month_data[:, 2])   # Column 2: Mean Tmin

            monthly_data.append([precip_sum, tmax_avg, tmin_avg])

    # Convert to 2D array of shape (z * 12, 4)
    weather_monthly = np.array(monthly_data) # Shape: (num_years * 12, 3)

    # Construct full matrix with time columns
    # Order: [Days, Month, Year, WaterYear, Precip, Tmax, Tmin]
    monthly_matrix = np.column_stack([
        days_vec,
        months,
        years,
        water_years,
        weather_monthly[:, 0], # Precip
        weather_monthly[:, 1], # Tmax
        weather_monthly[:, 2]  # Tmin
    ])

    # Trim Jan-Sep from first year (first 9 rows) and Oct-Dec from last year (last 3 rows)
    monthly_matrix = monthly_matrix[9:-3]

    # 5. Save CSV file
    np.savetxt(
        config.WEATHER_DIR / f"monthly_weather_set_{n}.csv",
        monthly_matrix,
        delimiter=",",
        fmt=["%d", "%d", "%d", "%d", "%f", "%f", "%f"],  # Ints for time, Floats for weather
        header="Days,Month,Year,WY,Prcp_mm,Tmax_C,Tmin_C",
        comments=""
    )

    # ---
    # Build input dataset for the water systems simulation model

    # Flatten the array into a continuous time series
    flat_set = leap_mat[n].reshape(-1, num_vars)

    # Drop rows where any column is NaN
    valid_days_mask = ~np.isnan(flat_set).any(axis=1)
    clean_set = flat_set[valid_days_mask]

    # Trim clean_set (shape: total_days, num_vars)
    water_year_set = clean_set[start_trim : len(clean_set) - end_trim]

    # save set as csv
    np.savetxt(config.WEATHER_DIR / f"weather_set_{n}.csv", 
               water_year_set[:, [0, -1]], delimiter=",", fmt="%f",
               header="precip_mm,evap_mm", comments="")


