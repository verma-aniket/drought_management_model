# Import base libraries
import numpy as np
import pandas as pd
import time

# Import repo custom libraries
from src.utils.data_loader import DataLoader
from src.utils import config
from src.utils.climate_functions import get_spi_month, rolling_sum_nan, identify_drought_events
from src.simulation import WaterBalanceModel

# --- Step 1. Read Input Data ---
print("🧪 Initializing Water Balance Model Quick Test...")

# Instantiate data loader pipeline
loader = DataLoader()

# Load in the input data files
date_df          = loader.load_dates(loader.test_date_file)
weather_df       = loader.load_daily_timeseries(loader.weather_dir, loader.test_weather_file, len(date_df), cols=["precip_mm", "evap_mm"])
indicator_df     = loader.load_monthly_data(loader.weather_monthly_dir, loader.test_weather_monthly_file)
flow_df          = loader.load_daily_timeseries(loader.flow_dir, loader.test_flow_file, len(date_df), cols=["newell","bigtrees","tait","liddel","laguna","majors"])
demand_df        = loader.load_monthly_data(loader.demand_dir, loader.test_demand_file)
params_dict      = loader.load_flat_parameters(loader.test_parameters_file)
bathymetry_df    = loader.load_stage_storage_area(loader.test_bathymetry_file)
profiles_dict    = loader.load_monthly_profiles(loader.test_profiles_file)
hydro_types_dict = loader.load_hydro_types(loader.test_hydro_types_file)
env_flows_dict   = loader.load_env_flows(loader.test_env_flows_file)
spi_params = loader.load_indicator_params(loader.test_spi_file, cols=['a', 'scale', 'q'])

# --- Step 2. Build exogenous monthly indicator vector (SPI) ---
print("⚠️ Building exogenous drought indicator vector...")

# define SPI sum-window - MUST MATCH SPI parameters
k = 12
# k-1 here because spi vector indicates spi of previous month
precip_k12 = rolling_sum_nan(indicator_df['Prcp_mm'].values, k)[k-1:-1] # earliest data in indicator_df data should be least one year prior to simulation start date
spi = np.zeros(precip_k12.shape[0])
# build month vector
start_month = indicator_df.iloc[0]['month'] - 1 # minus one because first value in precip_k12 is for month prior
month_vector = ((start_month - 1 + np.arange(precip_k12.shape[0])) % 12 + 1).astype(np.int16)
for i in range(spi.shape[0]):
    # -1 to month to align with spi_param 0-index months (i.e., Jan = 0, Dec = 11)
    spi[i] = get_spi_month(precip_k12[i], spi_params[month_vector[i]-1])

# identify drought events
drought_vector = identify_drought_events(spi=spi, k1=0, k2=-1.5, m=6)

# --- Step 4. Define Policy Parameters ---
# define curtailment actions and demand hardening factor vectors
# Note: first action is "do nothing" which means no curtailment, so it does not have a demand hardening factor
curtail_action = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50] 
hardening_factor = [0.0, 0.50, 0.50, 0.50, 0.50, 0.50] # 50% demand hardening for each curtailment action
policy_thresholds = [1300, 1200, 1250, 1100, 1070]

# define curtailment cost factor
curtailment_factor = 1.0

# --- Step 5. Instantiate Model Instance
print("🏗️ Instantiating WaterBalanceModel...")
model = WaterBalanceModel(
    curt_rates=curtail_action,
    hardening_factor=hardening_factor,
    curt_weight=curtailment_factor,
    params_dict=params_dict, 
    profiles_dict=profiles_dict,
    hydro_types_dict=hydro_types_dict, 
    bathymetry_df=bathymetry_df, 
    env_flows_dict=env_flows_dict,
    date_df=date_df, 
    flow_df=flow_df, 
    weather_df=weather_df, 
    demand_df=demand_df,
    drought_status=drought_vector
)

# --- Step 6. Execute Simulation ---
print("🏃 Running water resources systems model simulation...")
start = time.perf_counter()
results_df = model.run_simulation(indicator_thresholds=policy_thresholds, results="operations")
stop = time.perf_counter()
duration = stop - start

# --- Step 7. Summary Results Display ---
print("\n📊 --- Simulation Key Values ---")
print(f"Total Simulation Days Processed: {len(results_df)}")
print(f"Final Newell Bucket Storage: {results_df['V_newell'].iloc[-1]:.2f} MG")
print(f"Final Felton Bucket Storage: {results_df['V_felton'].iloc[-1]:.2f} MG")
print(f"Total Cumulative Urban Shortfalls: {results_df['unmet_urban_demand'].sum():.2f} MG")
print(f"Total Curtailed Demand: {(results_df['base_demand_MGD'] - results_df['active_urban_demand']).sum():.2f} MG")
print(f"Execution time: {duration:.6f} seconds")
print("🎉 Test execution complete!")

# Optionally, save results to csv
results_df.to_csv(config.RESULTS_DIR / "test_results.csv", index=True, sep=',')

# End of Script