# plot SWG weather time series and sub-select relevant drought scenarios for initial results

# import root libraries
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# link core folders
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils import config
from src.utils import climate_functions
from src.utils.data_loader import DataLoader
from src.utils.plotting import plot_format
loader = DataLoader()

# set standard plot format
plt.rcParams['mathtext.fontset'] = 'cm'
# plt.rcParams['font.family'] = 'monospace'
plt.rcParams['xtick.labelsize'] = 14
plt.rcParams['ytick.labelsize'] = 14
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['legend.fontsize'] = 14
text_font = 12

# read spi parameters data
spi_params = loader.load_indicator_params(config.SYSTEM_DIR / "spi_params.csv", cols=['a', 'scale', 'q'])

# parse SPI-based drought identification variables (based on run theory)
drought_clip = 0      # check for negative SPI
drought_spi = -1      # drought event must achive a minimum of at least -1
min_duration = 6      # SPI must be negative for 6 consecutive months

# define historic precipitaiton data vector
precip_ini = np.array([3, 127, 94.1, 196.9, 129.1, 178, 21.5, 82.9, 3.7, 0, 0.9, 4.2]) # total monthly precip in WY 2019 in mm

# compute number of droughts in sampled drought scenario indices
weather_sets = 10

# define empty matrices
start_year = 1937
stop_year = 2015
num_year = stop_year - start_year + 1
num_mon = num_year*12
spi_mat = np.zeros(shape=(weather_sets, num_mon))
drought_mat = np.zeros(shape=(weather_sets, num_mon))

for set in range(weather_sets):
    # read precipitation data
    precip = pd.read_csv(config.WEATHER_MONTHLY_DIR / f"weather_set_{set}.csv", usecols=['Prcp_mm']).to_numpy().reshape(-1)

    # compute number of droughts using SPI
    spi = climate_functions.calculate_spi(np.concatenate((precip_ini, precip)), spi_params)
    drought = climate_functions.identify_drought_events(spi, drought_clip, drought_spi, min_duration)
    num_d, time_d = climate_functions.get_num_time_drought(drought)
    print(f"Set: {set} has {num_d} drougths.")

    # store results for plotting
    spi_mat[set] = spi
    drought_mat[set] = drought

# plot spi and drought status of all 10 scenarios
years = np.arange(0,num_mon) / 12
for set in range(weather_sets):

    fig, ax = plt.subplots() # define a new figure

    ax.plot(years, spi_mat[set], lw=1, color = "#E04F39", zorder=5)
    ax.fill_between(years, -3*np.ones_like(years), 6*drought_mat[set]-3, color="#FEDD5C", alpha=0.5, zorder=4)

    # plot formatting
    plot_format(ax, "Years Since October 2019", "SPI", grid=True)
    ax.set_xlim([0,num_mon/12])
    ax.set_ylim([-3,3])
    ax.set_xticks(np.arange(0,(num_mon+1)/12,10))
    ax.set_yticks(np.arange(-3.0,0.2,3.1))

plt.show()