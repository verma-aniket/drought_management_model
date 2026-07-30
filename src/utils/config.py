import sys
from pathlib import Path

# 1. Dynamically locate the absolute root directory of this repository.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent # climb three levels to get to the main repo folder 

# Append repo root and 'src' folder to the Python system path.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Define 'src' folder path
SRC_DIR = REPO_ROOT / "src"

# Explicitly define relative subdirectories for data structures
DATA_DIR = REPO_ROOT / "data"
SYSTEM_DIR = DATA_DIR / "system"
WEATHER_DIR = DATA_DIR / "weather"
RAW_WEATHER_DIR = DATA_DIR / "raw_weather"
WEATHER_MONTHLY_DIR = DATA_DIR / "weather_monthly"
INDICATOR_DIR = DATA_DIR / "indicators"
FLOW_DIR = DATA_DIR / "flows"
DEMAND_DIR = DATA_DIR / "demands"
RESULTS_DIR = REPO_ROOT / "results"

# Define test filenames
TEST_DATE_FILE = "date_test.csv"
TEST_WEATHER_FILE = "weather_test.csv"
TEST_WEATHER_MONTHLY_FILE = "weather_test.csv"
TEST_INDICATOR_FILE = "indicator_test.csv"
TEST_FLOW_FILE = "flow_test.csv"
TEST_DEMAND_FILE = "demand_test.csv"
TEST_PARAM_FILE = "parameters.csv"
TEST_PROFILE_FILE = "monthly_profiles.csv"
TEST_BATHYM_FILE = "stage_storage_area.csv"
TEST_HYDRO_TYPE_FILE = "hydro_types.csv"
TEST_ENV_FLOWS_FILE = "reservoir_env_flows.csv"
TEST_SPI_FILE = "spi_params.csv"
TEST_DATA_YEAR_PRIOR_FILE = "data_year_prior.csv"

def initialize_directories():
    """
    Safely build the required workspace directories on the local machine 
    if they do not already exist when a simulation is executed.
    """
    directories = [DATA_DIR, SYSTEM_DIR, WEATHER_DIR, FLOW_DIR, DEMAND_DIR, RESULTS_DIR]
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    # Diagnostic test loop
    print("=== Repository Path Auto-Discovery ===")
    print(f"Repository Root: {REPO_ROOT}")
    print(f"Source Directory: {SRC_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"System Directory: {SYSTEM_DIR}")
    print(f"Weather Directory: {WEATHER_DIR}")
    print(f"Monthly Weather Directory: {WEATHER_MONTHLY_DIR}")
    print(f"Indicator Directory: {INDICATOR_DIR}")
    print(f"Flow Directory: {FLOW_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print("\n[+] Verifying and creating workspace folders...")
    initialize_directories()
    print("[✓] Path configuration verified successfully.")

# end of script