# TO DO LIST

# decided to run the hydro model using historic dates Jan 1937 to Sep 2015

# To align with 2019 as the start year, need to cut off 1937 and 1938
# trim first two years in output flows (flows start on October 1937, need to start at 1939)
# there are no leap years in this range so it's simple

# Task 1
# Run Hydro Model for all other scenarios

# Task 2
# A - trim years from processed daily output flow data
# B - trim first two years plus Jan to Sep of start year from daily weather input data
# C - define date text file starting at 2019  

# Task 3
# using monthly weather data (that starts in 1937), compute spi and trim off first two years
# will feed SPI as the drought indicator for the DPS model

# --------------------------------------------------------

# ABOVE 3 TASKS COMPLETE AND TEST RAN SUCCESFULLY FOR SET 0

# SET 1+ are all different lengths!!!!
# need to figure out why and fix it, they should all be the same length as set 0 (27759)

# also make sure drought scenario indexing logic is working as intended

# reads raw WeaGETS output and reformats it to pass into Santa Cruz rainfall-runoff model
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# Dynamically locate the absolute root directory of this repository.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent # climb three levels to get to the main repo folder

# Append repo root folder to the Python system path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# read config to access file strucutre
from src.utils import config
from src.utils.leap_year_functions import impute_leap_days, get_leap_year_indices

# Define SWG ID
swg_id = 1

# Days per month lookup
DAYS_NON_LEAP = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
DAYS_LEAP     = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# define mean daily percentage (p) of annual daytime hours for 37.5 deg latitude (Santa Cruz)
# took average of 35 and 40 deg value from this source: https://www.fao.org/4/s2022e/s2022e07.htm#3.1.3%20blaney%20criddle%20method
# based on what the rainfall runoff modeled used to compute daily reference evapotranspiration 
pr = {1: 0.225, 2: 0.245, 3: 0.27, 4: 0.295, 5: 0.315, 6: 0.33, 
      7: 0.325, 8: 0.305, 9: 0.28, 10: 0.25, 11: 0.225, 12: 0.215}

# obsolete
# # define simulation parameters
# start_year = 2020 # Jan 2020
# stop_year = 2069 # Dec 2069
# num_year = stop_year - start_year + 1 + 1 # first + 1 is for inclusivity, second +1 is because we will remove 1 year to align on water year

# for hydro-model date alignment
start_year = 1937
stop_year = 2015
num_year = stop_year - start_year + 1

# get files to loop over
files = [path for path in config.RAW_WEATHER_DIR.iterdir() if path.is_file()]

# read data into numpy array
mat = np.loadtxt(files[swg_id-1], delimiter=",")

# shave off extra years if needed
extra_years = (mat.shape[0] // 365) % num_year
if extra_years > 0:
    mat = mat[:-extra_years * 365]
num_days = mat.shape[0]

# add reference evaporation in mm/day to the matrix data using the Blaney-Criddle equation
month_vector = np.repeat(np.arange(1, 13), DAYS_NON_LEAP)  # Build 1D array of length 365: [1, 1... 1, 2, 2... 2, ..., 12, 12... 12, Shape: (365,)
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
num_sets = tot_years // (num_year)
mat = mat.reshape(num_sets, num_year, 365, num_vars)

# build output file

# impute leap days
leap_year_idx = get_leap_year_indices(start_year, num_year+1)
leap_mat = impute_leap_days(mat, leap_year_idx)

# Days to trim from the start of Year 1 (Jan 1 to Sep 30 = 273 days in non-leap, 274 in leap)
start_is_leap = (start_year % 4 == 0 and start_year % 100 != 0) or (start_year % 400 == 0)
start_trim = 274 if start_is_leap else 273
start_trim += 2*365 # add two years to the start trim to cut off two years to align with hydro-model

# Days to trim from the end of the final year (Oct 1 to Dec 31 = 92 days in both leap and non-leap)
# (October has 31 days, November 30, December 31 -> always 92 days total)
end_trim = 92

# define constants for daily to monthly matrix manipulation 

# Month slicing boundaries for 366-day padded year
month_slice_bounds = np.cumsum([0] + DAYS_LEAP)

# Define time vectors across the set length (e.g., 101 years)
years = np.repeat(np.arange(start_year, start_year + num_year), 12) 
months = np.tile(np.arange(1, 13), num_year)

# Generate 'Days' vector accounting for leap years
days_list = []
for yr in range(start_year, start_year + num_year):
    is_leap = (yr % 4 == 0) and (yr % 100 != 0 or yr % 400 == 0)
    days_list.extend(DAYS_LEAP if is_leap else DAYS_NON_LEAP)
days_vec = np.array(days_list)

# Generate Water Year vector: WY = Year + 1 for Oct-Dec (Months 10, 11, 12)
water_years = np.where(months >= 10, years + 1, years)

# save files based on selectively sampled weather series 
drought_idx = np.loadtxt(config.DATA_DIR / f"other/drought_idx_{swg_id}.txt", dtype=np.int32, delimiter=",")

# define counter
counter = 0

for n in drought_idx:

    # if counter>0:
    #     break
    
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

    # OBSOLETE, hydro model needs Jan to Sep of first year (it will be removed in the output anyways)
    # # Trim Jan-Sep from first year (first 9 rows) and Oct-Dec from last year (last 3 rows)
    # # monthly_matrix = monthly_matrix[9:-3] ### REMOVED THIS TO DEBUG HYDRO MODEL

    # # Trim Oct-Dec from last year (last 3 rows) to align with hydro model
    # monthly_matrix = monthly_matrix[0:-3]


    # Save CSV file
    np.savetxt(
        config.WEATHER_MONTHLY_DIR / f"weather_set_{counter}.csv",
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
    np.savetxt(config.WEATHER_DIR / f"weather_set_{counter}.csv", 
               water_year_set[:, [0, -1]], delimiter=",", fmt="%f",
               header="precip_mm,evap_mm", comments="")

    # increase counter
    counter += 1

# Build dates csv file
model_start_year = 2019
model_end_year = num_year + model_start_year - 1 - 2
date_vector = pd.date_range(start=f"{model_start_year}-10-01", end=f"{model_end_year}-09-30", freq="D")
date_df = pd.DataFrame(data={
    'year':  date_vector.year,
    'month': date_vector.month,
    'day':   date_vector.day,
})
date_df.to_csv(config.SYSTEM_DIR / f"dates.csv" , index=False, sep=',')

