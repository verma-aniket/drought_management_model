# plot PDF and CDF of daily and monthly preciptiation between orignial UMass simulation model and SWG
import sys
from pathlib import Path
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

# Dynamically locate the absolute root directory of this repository.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent # climb three levels to get to the main repo folder

if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from src.utils import config
from src.utils import climate_functions
from src.utils.data_loader import DataLoader
loader = DataLoader()

# read model daily and monthly precip data
model_daily_dates = loader.load_dates("date_test.csv")
model_daily_P = loader.load_daily_timeseries(config.WEATHER_DIR, "weather_test.csv", len(model_daily_dates), cols=['precip_mm'])
model_daily_P['Month'] = model_daily_dates["month"].values
model_monthly_P = pd.read_csv(config.WEATHER_MONTHLY_DIR / "weather_test.csv", usecols=["Month", "Prcp_mm"])

# read and process SWG daily and monthly precip data
swg_id = 3
daily_P = np.loadtxt(config.RAW_WEATHER_DIR / f"SWG_P2_{swg_id}.csv", delimiter=",", usecols=[0])
num_days = daily_P.shape[0]
daily_P = daily_P.reshape(num_days // 365, 365)
month_lengths = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
day_to_month = np.repeat(np.arange(0, 12), month_lengths)
monthy_P = np.zeros((daily_P.shape[0], 12))
for m in range(12):
    monthy_P[:, m] = daily_P[:, day_to_month == m].sum(axis=1)
daily_months = np.tile(day_to_month, daily_P.shape[0]) + 1
df_daily = pd.DataFrame({
    'precip_mm': daily_P.flatten(),
    'Month': daily_months
})
monthly_months = np.tile(np.arange(1, 13), monthy_P.shape[0])  # Shape: (N * 12,)
df_monthly = pd.DataFrame({
    'Prcp_mm': monthy_P.flatten(),
    'Month': monthly_months
})

# Daily Precipitation Plot
model_daily_P['source'] = 'Model'
df_daily['source'] = 'SWG'

# Combine into a single DataFrame
df_combined = pd.concat([model_daily_P, df_daily], ignore_index=True)
plt.figure(figsize=(12, 6))

# Boxplot comparing the two datasets by month
ax = sns.boxplot(
    data=df_combined,
    x='Month',
    y='precip_mm',
    hue='source',
    palette='Set2',
    showfliers=False
)

# ax = sns.violinplot(
#     data=df_combined,
#     x='Month',
#     y='precip_mm',
#     hue='source',
#     split=True,         
#     inner='quartile',   
#     palette='Set2',
#     cut=0               
# )

# Formatting
plt.xlabel('Month', fontsize=12)
plt.ylabel('Daily Precipitation (mm)', fontsize=12)
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Dataset', title_fontsize='11', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save Figure
plt.savefig(REPO_ROOT / 'plots/daily_P_compare.png', dpi=300, bbox_inches='tight')

# Daily Precipitation Plot
model_monthly_P['source'] = 'Model'
df_monthly['source'] = 'SWG'

# Combine into a single DataFrame
df_combined = pd.concat([model_monthly_P, df_monthly], ignore_index=True)
plt.figure(figsize=(12, 6))

# Boxplot comparing the two datasets by month
ax = sns.boxplot(
    data=df_combined,
    x='Month',
    y='Prcp_mm',
    hue='source',
    palette='Set2',
    showfliers=False
)

# ax = sns.violinplot(
#     data=df_combined,
#     x='Month',
#     y='Prcp_mm',
#     hue='source',
#     split=True,
#     inner='quartile',
#     palette='Set2',
#     cut=0
# )

# Formatting
plt.xlabel('Month', fontsize=12)
plt.ylabel('Monthly Precipitation (mm)', fontsize=12)
plt.xticks(ticks=range(12), labels=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                    'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.legend(title='Dataset', title_fontsize='11', loc='upper right')
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()

# Save Figure
plt.savefig(REPO_ROOT / 'plots/monthly_P_compare.png', dpi=300, bbox_inches='tight')