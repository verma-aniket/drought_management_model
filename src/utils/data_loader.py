import pandas as pd
from pathlib import Path
# from src.utils import config
try:
    from src.utils import config
except (ImportError, ModuleNotFoundError):
    import config  # Fallback for when running the file directly from inside /src

class DataLoader:
    def __init__(self):
        """
        Initializes the data pipeline manager using the path definitions in config.py
        """
        # Initialize directories
        self.system_dir = config.SYSTEM_DIR
        self.weather_dir = config.WEATHER_DIR
        self.weather_monthly_dir = config.WEATHER_MONTHLY_DIR
        self.indicator_dir = config.INDICATOR_DIR
        self.flow_dir = config.FLOW_DIR
        self.demand_dir = config.DEMAND_DIR

        # Initialize test file names
        self.test_date_file = config.TEST_DATE_FILE
        self.test_flow_file = config.TEST_FLOW_FILE
        self.test_weather_file = config.TEST_WEATHER_FILE
        self.test_weather_monthly_file = config.TEST_WEATHER_MONTHLY_FILE
        self.test_demand_file = config.TEST_DEMAND_FILE
        self.test_parameters_file = config.TEST_PARAM_FILE
        self.test_hydro_types_file = config.TEST_HYDRO_TYPE_FILE
        self.test_bathymetry_file = config.TEST_BATHYM_FILE
        self.test_env_flows_file = config.TEST_ENV_FLOWS_FILE
        self.test_profiles_file = config.TEST_PROFILE_FILE
        self.test_spi_file = config.TEST_SPI_FILE
        self.test_data_year_prior_file = config.TEST_DATA_YEAR_PRIOR_FILE

    def load_dates(self, filename: str) -> pd.DataFrame:
        """
        Loads a daily time-series file (weather or flows), 
        combines year-month-day columns into a pandas DatetimeIndex and ensures chronological sorting.
        """
        file_path = self.system_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Dates data not found: {file_path}")
            
        # print(f"[-] Parsing dates from: {file_path.name}")
        df = pd.read_csv(file_path)
        
        required_date_cols = ['year', 'month', 'day']
        if not all(col in df.columns for col in required_date_cols):
            raise KeyError(f"CRITICAL: {filename} must contain 'year', 'month', and 'day' columns")
            
        # Combine and set unified datetime index
        df['date'] = pd.to_datetime(df[required_date_cols])
        df = df.set_index('date').sort_index()

        # add an index column (to be used for accessing values from turbidity vector)
        df['t'] = range(len(df)) 
        df['t'] = df['t'].astype(int)

        return df

    def load_daily_timeseries(self, folder_path: Path, filename: str, required_length: int, cols: list = None) -> pd.DataFrame:
        """
        Loads a daily time-series file (weather or flows), 
        combines year-month-day columns into a pandas DatetimeIndex and ensures chronological sorting.
        required_length is the total number of daily entries that must be in the data.
        """
        file_path = folder_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Time-series data not found: {file_path}")
            
        # print(f"[-] Parsing daily timeline metrics from: {file_path.name}")
        if cols is None:
            df = pd.read_csv(file_path)
        else:
            df = pd.read_csv(file_path, usecols=cols)
        
        if len(df) != required_length:
            raise KeyError(f"CRITICAL: {filename} must be aligned with the dates input data.")
        
        # required_date_cols = ['year', 'month', 'day']
        # if not all(col in df.columns for col in required_date_cols):
        #     raise KeyError(f"CRITICAL: {filename} must contain 'year', 'month', and 'day' columns")
            
        # # Combine and set unified datetime index
        # df['date'] = pd.to_datetime(df[required_date_cols])
        # df = df.set_index('date').sort_index()

        # # add an index column (to be used for accessing values from turbidity vector)
        # df['t'] = range(len(df)) 
        # df['t'] = df['t'].astype(int)

        return df

    def load_monthly_data(self, folder_path: Path, filename: str) -> pd.DataFrame:
        """
        Loads a monthly demand projection, 
        combines year-month columns into a pandas DatetimeIndex and ensures chronological sorting.
        """
        file_path = folder_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Demand projections not found: {file_path}")
            
        # print(f"[-] Parsing monthly demand projections from: {file_path.name}")
        df = pd.read_csv(file_path)

        # lower all column name characters
        required_date_cols = ['year', 'month']
        df.columns = [col.lower() if col.lower() in required_date_cols else col for col in df.columns] 
        
        if not all(col in df.columns for col in required_date_cols):
            raise KeyError(f"CRITICAL: {filename} must contain 'year' and 'month' columns")
        
        # Create a temporary series representing the 1st day of that year-month
        # This converts integers/strings into a standard "YYYY-MM-01" format
        date_strings = df['year'].astype(str) + '-' + df['month'].astype(str).str.zfill(2) + '-01'

        # Combine and set unified datetime index
        df['date'] = pd.to_datetime(date_strings)
        df = df.set_index('date').sort_index().reset_index(drop=True)
        
        return df

    def load_flat_parameters(self, filename: str = "parameters.csv") -> dict:
        """
        Reads a 4-column flat parameter file (key, value, units, description).
        Maps 'key' to 'value' into a dictionary, ignoring metadata columns.
        """
        file_path = self.system_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Parameter file not found: {file_path}")
            
        # print(f"[-] Compiling flat system parameters from: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # Extract the first two columns (key and value)
        key_col = df.columns[0]
        val_col = df.columns[1]
        
        return dict(zip(df[key_col].astype(str).str.strip(), df[val_col]))

    def load_stage_storage_area(self, filename: str = "stage_storage_area.csv") -> pd.DataFrame:
        """
        Loads the reservoir bathymetry matrix mapping stage_ft_NCDD, storage_MG, and area_acre.
        Returns a sorted DataFrame used for linear interpolation down the line.
        Notes: 
        1. NCDD = Newell Creek Dam Datum, is a highly localized historical reference system used in Santa Cruz
        2. NAVD88 = North American Vertical Datum of 1988, is the broader, modern standard used across the US
        """
        file_path = self.system_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Bathymetry file not found: {file_path}")
            
        # print(f"[-] Loading reservoir stage-storage-area matrix: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # Ensure it's sorted by storage volume for clean mathematical interpolation lookup
        if 'storage_MG' in df.columns:
            df = df.sort_values(by='storage_MG').reset_index(drop=True)
        return df

    def load_monthly_profiles(self, filename: str = "monthly_profiles.csv") -> dict:
        """
        Parses a profile file containing columns: key, jan, feb, ..., dec, units, description.
        Transforms it into a nested lookup dictionary: { 'key': { 1: jan_val, 2: feb_val, ... } }
        """
        file_path = self.system_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Monthly profiles file not found: {file_path}")
            
        # print(f"[-] Parsing monthly cyclic profiles from: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # Standardize month parsing mapping text columns to integer month numbers (1-12)
        month_cols = ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
        
        profiles_dict = {}
        for _, row in df.iterrows():
            key = str(row.iloc[0]).strip()
            # Map month index 1 to 'jan', 2 to 'feb', etc.
            profiles_dict[key] = {m_idx + 1: row[m_col] for m_idx, m_col in enumerate(month_cols)}
            
        return profiles_dict
    
    def load_hydro_types(self, filename: str = "hydro_types.csv") -> dict:
        """
        Loads the water-year cumulative volume thresholds table.
        Converts text months (oct, nov, etc.) into integer month keys (10, 11, etc.)
        and maps them to a dictionary of thresholds for rapid execution lookups.
        """
        file_path = self.system_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Hydrologic type file not found: {file_path}")
            
        # print(f"[-] Parsing hydrologic condition classification parameters from: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # Mapping dictionary to match lowercased column names to integer calendar months
        month_str_to_int = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        hydro_dict = {}
        # Columns we expect to extract thresholds for
        condition_cols = ['driest', 'dry', 'normal', 'wet', 'wettest']
        
        # Detect the name of the first column (e.g., 'month')
        month_col_name = df.columns[0]
        
        for _, row in df.iterrows():
            raw_month = str(row[month_col_name]).strip().lower()[:3] # Normalize to 3-char string
            month_int = month_str_to_int.get(raw_month)
            
            if month_int is None:
                raise ValueError(f"CRITICAL: Unrecognized month value '{row[month_col_name]}' in {filename}")
                
            # Store threshold limits mapped directly under the month integer identifier
            hydro_dict[month_int] = {col: float(row[col]) for col in condition_cols if col in df.columns}
            
        return hydro_dict
    
    def load_env_flows(self, filename: str = "reservoir_env_flows.csv") -> dict:
        """
        Loads the LL reservoir environmental spill minimum flows table.
        Converts text months (jan, feb, etc.) into integer month keys (1, 2, etc.)
        and maps them to a dictionary of threshold, low_flow, and high_flow for rapid execution lookups.
        """
        file_path = self.system_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Minimum environmental spills file not found: {file_path}")
            
        # print(f"[-] Parsing reservoir minimum environmental spills data from: {file_path.name}")
        df = pd.read_csv(file_path)
        
        # Mapping dictionary to match lowercased column names to integer calendar months
        month_str_to_int = {
            'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
            'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        }
        
        env_flow_dict = {}
        # Columns we expect to extract data from
        condition_cols = ['threshold_MG', 'low_flow_cfs', 'high_flow_cfs']
        
        # Detect the name of the first column (e.g., 'month')
        month_col_name = df.columns[0]
        
        for _, row in df.iterrows():
            raw_month = str(row[month_col_name]).strip().lower()[:3] # Normalize to 3-char string
            month_int = month_str_to_int.get(raw_month)
            
            if month_int is None:
                raise ValueError(f"CRITICAL: Unrecognized month value '{row[month_col_name]}' in {filename}")
                
            # Store column data mapped directly under the month integer identifier
            env_flow_dict[month_int] = [float(row[col]) for col in condition_cols if col in df.columns]
            
        return env_flow_dict

    def load_indicator_params(self, filename: str = "spi_params.csv", cols: list = ['a', 'scale', 'q']) -> pd.DataFrame:
        """
        Loads monthly indicator parameters (i.e., SPI, SRI, etc.)
        """
        file_path = self.indicator_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Indicator file not found: {file_path}")

        # df = pd.read_csv(file_path)

        # required_date_cols = ['month']
        # if not all(col in df.columns for col in required_date_cols):
        #     raise KeyError(f"CRITICAL: {filename} must contain 'month' column")
            
        # # Mapping dictionary to match lowercased column names to integer calendar months
        # month_str_to_int = {
        #     'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
        #     'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
        # }
        
        # spi_dict = {}
        
        # # Detect the name of the first column (e.g., 'month')
        # month_col_name = df.columns[0]
        
        # for _, row in df.iterrows():
        #     if isinstance(row[month_col_name], str):
        #         raw_month = str(row[month_col_name]).strip().lower()[:3] # Normalize to 3-char string
        #         month_int = month_str_to_int.get(raw_month)
        #     else:
        #         month_int = int(row[month_col_name])
            
        #     if month_int is None:
        #         raise ValueError(f"CRITICAL: Unrecognized month value '{row[month_col_name]}' in {filename}")
                
        #     # Store column data mapped directly under the month integer identifier
        #     spi_dict[month_int] = [float(row[col]) for col in cols if col in df.columns]

        arr = pd.read_csv(file_path, usecols=cols).to_numpy()  

        return arr 

if __name__ == "__main__":
    print("=== Input Data Loading Test ===")
    
    # 1. Instantiate loader pipeline
    loader = DataLoader()
    
    # 2. Run sequential extraction loop for all input data files
    try:
        dates_df = loader.load_dates("date_test.csv")
        num_days = len(dates_df)
        weather_df = loader.load_daily_timeseries(loader.weather_dir, "weather_test.csv", num_days, cols=['precip_mm', 'evap_mm'])
        monthly_weather_df = loader.load_monthly_data(loader.weather_monthly_dir, "weather_test.csv")
        data_year_prior_df = loader.load_monthly_data(loader.indicator_dir, "data_year_prior.csv")
        flows_df   = loader.load_daily_timeseries(loader.flow_dir, "flow_test.csv", num_days)
        demand_df  = loader.load_monthly_data(loader.demand_dir, "demand_test.csv")
        params = loader.load_flat_parameters("parameters.csv")
        bathymetry  = loader.load_stage_storage_area("stage_storage_area.csv")
        profiles    = loader.load_monthly_profiles("monthly_profiles.csv")
        hydro_types = loader.load_hydro_types("hydro_types.csv")
        env_flows = loader.load_env_flows("reservoir_env_flows.csv")
        spi_params = loader.load_indicator_params("spi_params.csv", cols=['a', 'scale', 'q'])

        print("\n" + "="*50)
        print("ALL FILE ASSETS SUCCESSFULLY VERIFIED AND PARSED!")
        print("="*50)
        print(f" -> Date Timeline:         {dates_df.index.min().date()} to {dates_df.index.max().date()} ({len(dates_df)} days)")
        print(f" -> Weather Data:          {len(weather_df)} days of weather data loaded.")
        print(f" -> Monthly Weather Data:  {len(monthly_weather_df)} months of weather data loaded.")
        print(f" -> Year Prior Data:       {len(data_year_prior_df)} months of previous year data loaded.")
        print(f" -> Flow Data Timeline:    {len(flows_df)} days of flow data loaded.")
        print(f" -> Demand Data Entries:   {len(demand_df)} monthly periods mapped.")
        print(f" -> System Parameters:     {len(params)} core configuration values loaded.")
        print(f" -> Bathymetry Rows:       {len(bathymetry)} lines parsed from curves.")
        print(f" -> Profile Types:         {list(profiles.keys())}")
        print(f" -> October 'Normal' Vol:  {hydro_types[10]['normal']} (Month 10 verified)")
        print(f" -> Feburary Threshold:    {env_flows[2][0]} (Month 2 verified)")
        print(f" -> June Parameters:       {spi_params[5]} (Month 6 verified)")
        print("="*50)
        
    except FileNotFoundError as fnf:
        print(f"\n File Resolution Error: {fnf}")
        print("     Please ensure files exist in your local storage paths exactly as specified.")
    except Exception as e:
        print(f"\n Unexpected Data Processing Failure: {e}")