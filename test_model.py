import numpy as np
import pandas as pd
from src.utils.data_loader import DataLoader
from src.utils import config
from src.utils.climate_functions import get_spi_month, rolling_sum_nan
from src.simulation import WaterBalanceModel

import time


# Debugging steps

# the turbidity condition has a greater impact than I anticpiated, need a way to resolve this

def run_pipeline_test():

    # define objective function weights
    deficit_weight = 0.25
    curt_cost_weight = 0.25
    HR2W_penalty = 0.5
    
    # print("🧪 Initializing Water Balance Model Quick Test...")
    loader = DataLoader() # instantiate loader pipeline

    # --- 1. Load in the input data files ---
    # print("📖 Loading test data files...")
    date_df          = loader.load_dates(loader.test_date_file)
    weather_df       = loader.load_daily_timeseries(loader.weather_dir, loader.test_weather_file, len(date_df))
    monthly_weather_df = loader.load_monthly_data(loader.weather_monthly_dir, loader.test_weather_monthly_file)
    data_year_prior_df = loader.load_monthly_data(loader.indicator_dir, loader.test_data_year_prior_file)
    flow_df          = loader.load_daily_timeseries(loader.flow_dir, loader.test_flow_file, len(date_df))
    # date_df          = loader.load_dates("dates.csv")
    # weather_df       = loader.load_daily_timeseries(loader.weather_dir, "weather_set_0.csv", len(date_df))
    # flow_df          = loader.load_daily_timeseries(loader.flow_dir, "flow_set_0.csv", len(date_df))
    demand_df        = loader.load_monthly_data(loader.demand_dir, loader.test_demand_file)
    params_dict      = loader.load_flat_parameters(loader.test_parameters_file)
    bathymetry_df    = loader.load_stage_storage_area(loader.test_bathymetry_file)
    profiles_dict    = loader.load_monthly_profiles(loader.test_profiles_file)
    hydro_types_dict = loader.load_hydro_types(loader.test_hydro_types_file)
    env_flows_dict   = loader.load_env_flows(loader.test_env_flows_file)
    spi_params       = loader.load_indicator_params(loader.test_spi_file, cols=['a', 'scale', 'q'])

    # --- 2. Build exogenous monthly indicator vectors (SPI or SRI) ---
    k = 12
    # k-1 here because spi vector indicates spi of previous month
    precip_k12 = rolling_sum_nan(np.concatenate((data_year_prior_df['Prcp_mm'].values, monthly_weather_df['Prcp_mm'].values)), k)[k-1:-1]
    spi = np.zeros(precip_k12.shape[0])
    # build month vector
    start_month = data_year_prior_df.iloc[0]['month'] - 1 # minus one because first value in precip_k12 is for month prior
    month_vector = ((start_month - 1 + np.arange(precip_k12.shape[0])) % 12 + 1).astype(np.int16)
    for i in range(spi.shape[0]):
        # -1 to month to align with spi_param 0-index months (i.e., Jan = 0, Dec = 11)
        spi[i] = get_spi_month(precip_k12[i], spi_params[month_vector[i]-1])

    # --- 4. Define Policy Parameters ---
    # policy_thresholds = [1.5567, -0.6228, -1.8976, -2.5357, -3.0000]
    # policy_thresholds = [1.9073, 1.0210, 0.1631, -0.6338, -2.6244]
    policy_thresholds = [-1.0, -1.5, -2.0, -2.5, -3.0]
    policy_thresholds = [2.9417, 1.2214, 1.1320, -1.0232, -1.3397]
    curtail_action = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50]
    hardening_factor = 0.25


    # --- 5. Initialize Model Instance ---
    # print("🏗️ Instantiating WaterBalanceModel...")
    model = WaterBalanceModel(
        curt_rates=curtail_action,
        hardening_factor=hardening_factor,
        def_w = deficit_weight,
        curt_cost_w=curt_cost_weight,
        penalty_w=HR2W_penalty,
        params_dict=params_dict,
        profiles_dict=profiles_dict,
        hydro_types_dict=hydro_types_dict,
        bathymetry_df=bathymetry_df,
        env_flows_dict=env_flows_dict,
        date_df=date_df,
        flow_df=flow_df,
        weather_df=weather_df,
        demand_df=demand_df,
        exogenous_indicator=spi
    )

    # --- 3. Execute Simulation ---
    start = time.perf_counter()
    output = model.run_simulation(indicator_thresholds=policy_thresholds, results="objective")
    stop = time.perf_counter()
    duration = stop - start
    # # --- 4. Quick Verification Sanity Checks ---
    # print("\n📊 --- Verification Sanity Checks ---")
    # print(f"Total Simulation Days Processed: {len(results_df)}")
    # print(f"Final Newell Bucket Storage: {results_df['V_newell'].iloc[-1]:.2f} MG")
    # print(f"Final Felton Bucket Storage: {results_df['V_felton'].iloc[-1]:.2f} MG")
    # print(f"Total Cumulative Urban Shortfalls: {results_df['unmet_urban_demand'].sum():.2f} MG")
    # print(f"Execution time: {duration:.6f} seconds")
    # print("🎉 Test execution complete!")

    # # save results to csv
    # output.to_csv(config.RESULTS_DIR / "test_results_demand.csv", index=True, sep=',')
    print(output) # 14934.350519

    output = model.run_simulation(indicator_thresholds=policy_thresholds, results="objective")

    print(output)

if __name__ == "__main__":
    run_pipeline_test()