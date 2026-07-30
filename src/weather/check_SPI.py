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
spi_params = loader.load_indicator_params("spi_params.csv", cols=['a', 'scale', 'q'])

# read raw SWG data - precipitation only, SWG_P2_2 has a scenario with 7 droughts!
precip = np.loadtxt(config.WEATHER_MONTHLY_DIR / f"weather_test.csv", delimiter=",", usecols=[4], skiprows=1)

# trim last 9 months so we start in January
precip = precip[:-9]

# define historic precipitaiton data vector (Jan 2019 to Sep 2019 - assumed to be monthly averages)
precip_ini = np.array([247.4113, 197.8452, 135.4207, 67.3871, 22.5375, 2.8033, 0.306, 2.41726, 12.66018]) 
precip_cat = np.concatenate((precip_ini, precip))

spi_check = climate_functions.calculate_spi(precip_cat, spi_params)

# december of first year
spi_test = climate_functions.get_spi_month(sum(precip_cat[0:12]), spi_params[11])

print(np.mean(spi_check))
print(np.std(spi_check))
print(np.min(spi_check))
print(np.max(spi_check))

