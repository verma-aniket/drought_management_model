import time
import numpy as np
import pandas as pd
from multiprocessing import Pool
from platypus import Problem, Real, GDE3, PoolEvaluator

# =====================================================================
# 1. WORKER INITIALIZATION (LOAD DATA & BUILD MODEL ONCE PER PROCESS)
# =====================================================================

# Global variable residing in each worker process's memory context
_global_model = None

def init_worker():
    """
    Executes ONCE in each worker process when the process pool is spawned.
    Loads input data files and builds the WaterBalanceModel locally.
    """
    global _global_model
    
    # Imports inside worker context
    from src.utils.data_loader import DataLoader
    from src.utils.climate_functions import get_spi_month, rolling_sum_nan
    from src.simulation import WaterBalanceModel

    # Define objective weights
    deficit_weight = 1
    curt_cost_weight = 0
    HR2W_penalty = 0

    # Load input datasets
    loader = DataLoader()
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

    # Compute SPI indicator series
    k = 12
    precip_k12 = rolling_sum_nan(
        np.concatenate((data_year_prior['Prcp_mm'].values, month_weather_df['Prcp_mm'].values)), k
    )[k-1:-1]
    
    spi = np.zeros(precip_k12.shape[0])
    start_month = data_year_prior.iloc[0]['month'] - 1
    month_vector = ((start_month - 1 + np.arange(precip_k12.shape[0])) % 12 + 1).astype(np.int16)
    
    for i in range(spi.shape[0]):
        spi[i] = get_spi_month(precip_k12[i], spi_params[month_vector[i] - 1])

    curtail_action = [0.0, 0.1, 0.20, 0.3, 0.4, 0.50]
    hardening_factor = 0.0

    # Assign initialized model to global worker instance
    _global_model = WaterBalanceModel(
        curt_rates=curtail_action,
        hardening_factor=hardening_factor,
        def_w=deficit_weight,
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

# =====================================================================
# 2. PLATYPUS PROBLEM WRAPPER
# =====================================================================

class WaterBalanceProblem(Problem):
    def __init__(self, k_bounds=(-3.0, 3.0), epsilon=0.01):

        super().__init__(5, 1, 4) # 5 variables, 1 objective (default is minimization), 4 constraints
        self.types[:] = Real(k_bounds[0], k_bounds[1])
        self.constraints[:] = "<=0"
        self.epsilon = epsilon

    def evaluate(self, solution):
        global _global_model
        K = solution.variables

        # 1. Evaluate objective function using process-local model instance
        # Platypus naturally minimizes this value
        solution.objectives[0] = _global_model.run_simulation(indicator_thresholds=K)

        # 2. Enforce K1 > K2 > K3 > K4 > K5 order constraints: (K_{i+1} - K_i + epsilon <= 0)
        solution.constraints[0] = K[1] - K[0] + self.epsilon
        solution.constraints[1] = K[2] - K[1] + self.epsilon
        solution.constraints[2] = K[3] - K[2] + self.epsilon
        solution.constraints[3] = K[4] - K[3] + self.epsilon

# =====================================================================
# 3. MAIN PARALLEL EXECUTION
# =====================================================================

if __name__ == "__main__":
    
    NUM_CORES = 1          # Parallel CPU processes
    POPULATION_SIZE = 50   # Candidates per generation
    N_EVALUATIONS = 100   # Total simulation evaluations

    print(f"⚙️ Spawning {NUM_CORES} CPU processes and loading model data into memory...")
    
    # Initialize process pool with init_worker function
    pool = Pool(processes=NUM_CORES, initializer=init_worker)

    try:
        evaluator = PoolEvaluator(pool)
        
        problem = WaterBalanceProblem(k_bounds=(-3.0, 3.0), epsilon=0.01)
        algorithm = GDE3(problem, population_size=POPULATION_SIZE, evaluator=evaluator)

        print(f"🚀 Starting Parallel GDE3 Optimization ({N_EVALUATIONS} evaluations across {NUM_CORES} cores)...")
        start_time = time.time()

        algorithm.run(N_EVALUATIONS)

        elapsed = time.time() - start_time
        print(f"\n Optimization finished in {elapsed:.2f} seconds!")

        # Filter for feasible solutions where all ordering constraints (K1 > K2 > K3 > K4 > K5) were met
        feasible_solutions = [s for s in algorithm.result if s.feasible]

        if feasible_solutions:
            best_solution = min(feasible_solutions, key=lambda s: s.objectives[0])
            
            print("\n" + "="*45)
            print("🎯 OPTIMAL POLICY THRESHOLDS FOUND")
            print("="*45)
            print(f"Best Objective Score: {best_solution.objectives[0]:.6f}")
            print("\nOptimal Thresholds:")
            for i, val in enumerate(best_solution.variables, start=1):
                print(f"  K{i}: {val:+.4f}")
            print("="*45)
        else:
            print("\n⚠️ No feasible solution satisfied the strict threshold ordering constraints.")

    finally:
        pool.close()
        pool.join()