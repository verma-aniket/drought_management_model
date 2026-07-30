# plot SWG weather time series and sub-select relevant drought scenarios for initial results

# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# link core folders
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils import config
from src.utils import climate_functions
from src.utils.data_loader import DataLoader
loader = DataLoader()

# Define SWG ID
swg_id = 1

# read spi parameters data
spi_params = loader.load_indicator_params(config.SYSTEM_DIR / "spi_params.csv", cols=['a', 'scale', 'q'])

# read raw SWG data - precipitation only, SWG_P2_2 has a scenario with 7 droughts!
precip_mat = np.loadtxt(config.RAW_WEATHER_DIR / f"SWG_P2_{swg_id}.csv", delimiter=",", usecols=[0])

# define historic precipitaiton data vector
precip_ini = np.array([196.9, 129.1, 178, 21.5, 82.9, 3.7, 0, 0.9, 4.2, 0.3, 70.6, 273.4]) # total monthly precip in 2019 in mm

# OBSOLETE
# # rehsape data into 50-year monthly precip matrix
# start_year = 2020 # Jan 2020
# stop_year = 2069 # Dec 2069
# num_years = stop_year - start_year + 1 + 1 # first + 1 is for inclusivity, second +1 is because we will remove 1 year to align on water year

# for hydro-model date alignment
start_year = 1937
stop_year = 2015
num_years = stop_year - start_year + 1

extra_years = (precip_mat.shape[0] // 365) % num_years
print(extra_years)
if extra_years > 0:
    precip_mat = precip_mat[:-extra_years * 365]
num_sets = (precip_mat.shape[0] // 365) // num_years
precip_mat = precip_mat.reshape(num_sets, num_years, 365)

# Build calendar mapping bounds (assuming standard non-leap year length blocks)
month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
day_to_month = np.repeat(np.arange(0, 12), month_lengths)

# Aggregate to monthly totals
precip_mon = np.zeros((precip_mat.shape[0], num_years, 12))
for m in range(12):
    precip_mon[:, :, m] = precip_mat[:, :, day_to_month == m].sum(axis=2)
precip_mon = precip_mon.reshape(precip_mat.shape[0], num_years * 12)

# BYPASS cutting one year off, start at January as usual
# # cut off first 9 months of first year and last 3 months of last year
# precip_mon = precip_mon[:, :-3]

# parse SPI-based drought identification variables (based on run theory)
drought_clip = 0      # check for negative SPI
drought_spi = -1      # drought event must achive a minimum of at least -1
min_duration = 6      # SPI must be negative for 6 consecutive months

num_droughts = np.zeros(precip_mon.shape[0])
years = np.repeat(np.arange(start_year, stop_year+1, 1), 12)

for i in range(precip_mon.shape[0]):
    # Call offloaded functions via the custom imported climate utils package
    spi = climate_functions.calculate_spi(np.concatenate((precip_ini, precip_mon[i])), spi_params)
    drought = climate_functions.identify_drought_events(spi, drought_clip, drought_spi, min_duration)
    num_d, time_d = climate_functions.get_num_time_drought(drought)

    # store number of droughts in current 51-year episode
    num_droughts[i] = num_d

# explore drought properties
print(f"{sum(num_droughts>0) / len(num_droughts)}% of 50-year records have at least one drought.")

print("Number of Droughts \t Record Count \t Fraction of Records \t ")
for i in range(0,int(max(num_droughts))+1):
    print(f"                 {i} \t {sum(num_droughts == i)} \t {sum(num_droughts == i)/len(num_droughts)}")

# random sampling here, size based on "Fraction of Records" above 
rng = np.random.default_rng(271124) # Initialize rng
d_6 = rng.choice(np.argwhere(num_droughts == 6).reshape(-1), size=1, replace=False)
d_5 = rng.choice(np.argwhere(num_droughts == 5).reshape(-1), size=1, replace=False)
d_4 = rng.choice(np.argwhere(num_droughts == 4).reshape(-1), size=1, replace=False)
d_3 = rng.choice(np.argwhere(num_droughts == 3).reshape(-1), size=2, replace=False)
d_2 = rng.choice(np.argwhere(num_droughts == 2).reshape(-1), size=2, replace=False)
d_1 = rng.choice(np.argwhere(num_droughts == 1).reshape(-1), size=2, replace=False)
d_0 = rng.choice(np.argwhere(num_droughts == 0).reshape(-1), size=1, replace=False)

# merge and save indicies
drought_idx = np.concat([d_0, d_1, d_2, d_3, d_4, d_5, d_6]).astype(np.int32)
np.savetxt(config.DATA_DIR / f"other/drought_idx_{swg_id}.txt", drought_idx, delimiter=",", fmt="%d")
