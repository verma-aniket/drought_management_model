import numpy as np
import pandas as pd
import time
from src.utils.data_loader import DataLoader
from src.utils import config
from src.utils.climate_functions import get_spi_month, rolling_sum_nan
from src.simulation import WaterBalanceModel
from platypus import Problem, Real, GDE3

# Define the platypus wrapper 
class WaterBalanceProblem(Problem):

  def __init__(self, model_instance, k_bounds=(-3.0, 3.0), epsilon=0.01):
    super().__init__(5, 1, 4)  # 5 vars, 1 objective, 4 constraints
    self.model = model_instance
    self.epsilon = epsilon

    self.types[:] = Real(k_bounds[0], k_bounds[1])
    self.constraints[:] = "<=0"

  def evaluate(self, solution):
    K = solution.variables

    # Run the physics model using candidate parameters
    solution.objectives[0] = self.model.run_simulation(
        indicator_thresholds=K)

    # Evaluate order constraints: K_{i+1} - K_i + epsilon <= 0
    solution.constraints[0] = K[1] - K[0] + self.epsilon
    solution.constraints[1] = K[2] - K[1] + self.epsilon
    solution.constraints[2] = K[3] - K[2] + self.epsilon
    solution.constraints[3] = K[4] - K[3] + self.epsilon

# --- Step 1: Build Simulation Model ---

# define objective function weights
deficit_weight = 0.25
curt_cost_weight = 0.25
HR2W_penalty = 0.5

# instantiate data loader pipeline
loader = DataLoader()

# Load in the input data files
date_df          = loader.load_dates(loader.test_date_file)
weather_df       = loader.load_daily_timeseries(loader.weather_dir, loader.test_weather_file, len(date_df))
month_weather_df = loader.load_monthly_data(loader.weather_monthly_dir, loader.test_weather_monthly_file)
data_year_prior  = loader.load_monthly_data(loader.indicator_dir, loader.test_data_year_prior_file)
flow_df          = loader.load_daily_timeseries(loader.flow_dir, loader.test_flow_file, len(date_df))
demand_df        = loader.load_monthly_data(loader.demand_dir, loader.test_demand_file)
params_dict      = loader.load_flat_parameters(loader.test_parameters_file)
bathymetry_df    = loader.load_stage_storage_area(loader.test_bathymetry_file)
profiles_dict    = loader.load_monthly_profiles(loader.test_profiles_file)
hydro_types_dict = loader.load_hydro_types(loader.test_hydro_types_file)
env_flows_dict   = loader.load_env_flows(loader.test_env_flows_file)
spi_params       = loader.load_indicator_params(loader.test_spi_file, cols=['a', 'scale', 'q'])

# Build exogenous monthly indicator vectors (SPI or SRI) 
k = 12
# k-1 here because spi vector indicates spi of previous month
precip_k12 = rolling_sum_nan(np.concatenate((data_year_prior['Prcp_mm'].values, month_weather_df['Prcp_mm'].values)), k)[k-1:-1]
spi = np.zeros(precip_k12.shape[0])
# build month vector
start_month = data_year_prior.iloc[0]['month'] - 1 # minus one because first value in precip_k12 is for month prior
month_vector = ((start_month - 1 + np.arange(precip_k12.shape[0])) % 12 + 1).astype(np.int16)
for i in range(spi.shape[0]):
    # -1 to month to align with spi_param 0-index months (i.e., Jan = 0, Dec = 11)
    spi[i] = get_spi_month(precip_k12[i], spi_params[month_vector[i]-1])

# Define Policy Parameters
curtail_action = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]
hardening_factor = 0.10

# Initialize Model Instance
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

# --- Step 2: Instantiate Problem & Optimizer ---

# 1. Create the optimization problem instance
problem = WaterBalanceProblem(model_instance=model, k_bounds=(-3.0, 3.0), epsilon=0.01)

# 2. Instantiate Differential Evolution
# Platypus uses standard defaults: CR=0.7, F=0.5
algorithm = GDE3(problem, population_size=50)

# --- Step 3: Run Optimization ---

n_evaluations = 100  # Total simulation runs (Population Size x Generations)

print(f"🚀 Starting Differential Evolution optimization ({n_evaluations} evaluations)...")
start_time = time.time()

# Run the optimization loop
algorithm.run(n_evaluations)

elapsed_time = time.time() - start_time
print(f" Optimization complete in {elapsed_time:.2f} seconds!")

# --- Step 4: Extract and Display Results ---

# Filter for feasible solutions (solutions that respected K1 > K2 > K3 > K4 > K5)
feasible_solutions = [s for s in algorithm.result if s.feasible]

if feasible_solutions:
    # Find the best solution among feasible ones (lowest objective score)
    best_solution = min(feasible_solutions, key=lambda s: s.objectives[0])
    
    print("\n" + "="*40)
    print("🎯 OPTIMAL POLICY THRESHOLDS FOUND")
    print("="*40)
    print(f"Best Objective Score: {best_solution.objectives[0]:.6f}")
    print("\nOptimal Thresholds:")
    for i, val in enumerate(best_solution.variables, start=1):
        print(f"  K{i}: {val:+.4f}")
    print("="*40)
else:
    print("\n⚠️ No feasible solution found that satisfied all constraints.")
    print("Consider increasing evaluation count or adjusting variable bounds.")

