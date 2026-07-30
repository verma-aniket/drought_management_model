import numpy as np
import pandas as pd
from tqdm import tqdm

def curtailment_policy(indicator_thresholds, indicator_val):
    """
    Lookup Function: Maps a continuous indicator (e.g. SPI) to a discrete state level (0 to 5).
    """
    K1, K2, K3, K4, K5 = indicator_thresholds
    if indicator_val >= K1:
        return 0
    elif indicator_val >= K2:
        return 1
    elif indicator_val >= K3:
        return 2
    elif indicator_val >= K4:
        return 3
    elif indicator_val >= K5:
        return 4
    else:
        return 5

class WaterBalanceModel:
    def __init__(self, curt_rates, hardening_factor,
                 def_w, curt_cost_w, penalty_w, 
                 params_dict, profiles_dict, hydro_types_dict, bathymetry_df, env_flows_dict,
                 date_df, flow_df, weather_df, demand_df,
                 exogenous_indicator):

        # Store policy parameters
        self.curtailment_rates = curt_rates                                     # curtailment actions assuming simple tree structure
        self.f_hardening = hardening_factor                                     # demand hardening factor

        # Store objective function weights
        # normalize weights if needed
        total_w = def_w + curt_cost_w + penalty_w
        self.def_w = def_w / total_w                                            # demand shortage deficit weight (0 - 1)
        self.curt_cost_w = curt_cost_w / total_w                                # curtailment cost weight
        self.penalty_w = penalty_w / total_w                                    # penalty of demand below HR2W
        self.HR2W = 55                                                          # Human Right to Water (gal per capita per day)

        # Store all parameters and data matrices
        self.params = params_dict
        self.profiles = profiles_dict
        self.hydro_types = hydro_types_dict
        self.bathymetry = bathymetry_df
        self.env_flows = env_flows_dict

        # store all dates as well as flow and weather time series data
        self.date = date_df                                                     # Input daily dates for efficient time keeping
        self.flow = flow_df                                                     # Input daily flow time series of all catchments, in MGD
        self.weather = weather_df                                               # Input daily weather time series of entire system
        self.demand = demand_df                                                 # Input monthly time series of urban demand, in MG per month

        # Unit conversion constants
        self.CFS_to_MGD = 0.646317                                              # cubic feet per second to million-gallons per day
        self.ft_to_mm = 304.8                                                   # feet to millimeter
        self.AF_to_MG = 0.325851                                                # acre-foot to million-gallons
        self.in_to_mm = 25.4                                                    # inches to mm

        # Centralized look-up lists and arrays
        # Days in month for non-leap year, Index 1 = Jan, Index 2 = Feb, etc. Dummy 0 used for clean calendar month matching
        self.days_in_months_base = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        # # Discrete Felton pump flow rates in MGD
        self.felton_pump_rates = np.array([4.4, 5.9, 7.3, 8.4, 9.6, 10.2, 10.8, 11.7, 12.5, 13.7]) * self.CFS_to_MGD

        # Map static parameters
        self.c_lid = float(params_dict.get('c_lid', 0.0))                       # Liddel Creek maximium diversion capacity, MGD
        self.c_lag = float(params_dict.get('c_lag', 0.0))                       # Laguna Creek maximium diversion capacity, MGD
        self.c_maj = float(params_dict.get('c_maj', 0.0))                       # Majors Creek maximium diversion capacity, MGD
        self.c_NC  = float(params_dict.get('c_NC', 0.0))                        # North Coast Pipe maximium diversion capacity, MGD
        self.d_NC_Farm = float(params_dict.get('D_NC_farm', 0.0))               # annual North Coast farmers demand, MG
        self.c_tait = float(params_dict.get('c_tait', 0.0))                     # Tait Street maximum diversion capacity, MGD
        self.c_tait_total = float(params_dict.get('c_tait_total', 0.0))         # Combined maximum diversion capacity of Tait Street and NC, MGD
        self.c_LL_release = float(params_dict.get('c_LL_release', 0.0))         # LL Reservoir to GHWTP pipe maximum capacity, MGD
        self.slr_env_buffer = float(params_dict.get('slr_env_buffer', 0.323))   # SLR minimum environmental flow buffer at Felton Diversion
        self.d_slv = float(params_dict.get('D_SLV', 0.0))                       # San Lorenzo Valley daily demand from LL reservoir, MGD

        # Map monthly profiles
        self.p_NC_farm = profiles_dict['p_NC_farm']                             # North Coast farmer demand monthly distribution, %
        self.c_tait_well = profiles_dict['c_tait_well']                         # Tait well maximum extraction, MGD
        self.c_b12 = profiles_dict['c_b12']                                     # Beltz 12 well maximum extraction, MGD
        self.c_oakwell = profiles_dict['c_oakwell']                             # Live Oak wells maximum extraction, MGD
        self.c_felton = profiles_dict['c_felton']                               # Felton Diversion pipeline to LL reservoir maximum capacity, MGD
        
        # Map water rights license limits
        self.WR_b12_limit = float(params_dict.get('WR_b12', 40.0))              # Beltz 12 Well Water Right (critically dry years only, resets Oct. 1st), MG
        self.WR_oakwell_limit = float(params_dict.get('WR_oakwell', 170.0))     # Live Oak Wells Water Right (resets Oct 1st), MG
        self.WR_felton_limit = float(params_dict.get('WR_felton', 977.0))       # SLR Felton Diversion Water Right (resets Sep 1st), MG
        self.WR_store_limit = float(params_dict.get('WR_store', 1825.0))        # Newell Creek Collection Water Right (resets Sep 1st), MG
        self.WR_release_limit = float(params_dict.get('WR_release', 1042.0))    # Newell Creek Diversion Water Right (resets Sep 1st), MG

        # Map curtailment policy persistence and recovery filters
        self.min_hold = int(params_dict.get('min_hold_months', 3))              # minimum number of months curtailment policy2 action is in place
        self.min_recovery = int(params_dict.get('min_recovery_months', 3))      # minimum number of months indicator should remain above "do nothing" threshold

        # Initialize rolling cumulative water rights ledgers (reset annually)
        self.WR_b12_accum = 0.0                       
        self.WR_oakwell_accum = 0.0                   
        self.WR_felton_accum = 0.0                    
        self.WR_store_accum = 0.0                     
        self.WR_release_accum = 0.0

        # Map reservoir properties
        self.LL_min = float(params_dict.get('LL_min', 944.969))                 # LL reservoir minimum strorage volume, MG
        self.LL_max = float(params_dict.get('LL_max', 2859.737))                # LL reservoir maximum strorage volume, MG
        self.LL_reserve = float(params_dict.get('LL_reserve', 1070.0))          # LL reservoir drought reserve volume, MG
        self.LL_threshold = float(params_dict.get('LL_threshold', 2092.642))    # LL reservoir January 1st threshold volume, MG

        # Initialize permanent state variables (carried over day-to-day)
        self.V_newell = float(params_dict.get('LL_initial', 2670.0))            # Newell Creek water in LL reservoir, MG (initilized to start of simulation storage)
        self.V_felton = 0.0                                                     # Felton Diversion water in LL reservoir, MG
        self.V_precip = 0.0                                                     # direct precipitation water in LL reservoir, MG

        # Hydrologic climate and condition status variables
        self.is_critically_dry_year = False                                     # critically dry if annual inflow from Big Trees is < 29000 AF
        self.crit_dry_threshold = 29000 * self.AF_to_MG                         # constant to check annual inflow against, in MG
        self.hydro_cond_month = 'normal'                                        # driest, dry, normal, wet, or wettest depending on cumulative SLR flow

        # Felton Pump status/conditions
        self.is_felton_pump_on = True                                           # checks if the Felton pump can be operated this year
        self.first_flush = True                                                 # checks if "first flush" has happened yet, resets every September, assume it has happened for initialization
        self.first_flush_counter = 0                                            # counter use to track first flush status
        self.first_flush_flow = float(params_dict.get('ff_flow', 100.0))        # minimum first flush flow threshold

        # Turbidity Flag parameters
        self.turb_flag = float(params_dict.get('turb_flag', 0.67))*self.in_to_mm# rainfall above which turbidity flag is raised, inches converted to mm
        self.slr_turb = np.zeros(len(self.weather), dtype=np.uint8)             # Turbidity flag for Felton and Tait Diversion
        self.nc_turb = np.zeros_like(self.slr_turb)                             # Turbidity flag for Liddel and Laguna Creek diversion
        self.maj_turb = np.zeros_like(self.slr_turb)                            # Turbidity flag for Majors Creek diversion

        # ENDOGENIZED DEMAND SETUP

        # Create a unique sequential key for every single month in the simulation timeline
        # This maps calendar (year, month) to a simple sequential integer index (0, 1, 2...)
        unique_months = self.date[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).reset_index(drop=True)

        # Map (year, month) -> sequential month index for instant internal lookups
        self.month_to_idx = {tuple(x): i for i, x in enumerate(unique_months.values)}

        # build baseline demand vector (known a priori given demand input) and population vector
        self.demand_base = np.zeros(len(unique_months))
        self.population = np.zeros(len(unique_months))

        for (yr, mn), idx in self.month_to_idx.items():
            # Grab original baseline projection
            base_val, pop = self.demand.loc[(self.demand['year'] == yr) & (self.demand['month'] == mn), ['demand_MG', 'population']].values[0]
            self.demand_base[idx] = base_val
            self.population[idx] = pop

        # # create daily population array
        # self.population = np.zeros(len(self.date))

        # # store exogenous indicator (i.e., SPI of previous month, assume year prior to simulation year is an average year, will fix this later)
        # self.indicator = np.zeros(len(self.date))

        # # build baseline demand vector (known a priori given demand input) and population and indicator vectors
        # # assumptions: daily demand is uniform given monthly demand (daily demand = monthly demand / no. of days in month)
        # self.demand_base = np.zeros(len(self.date))
        # unique_months = self.date[['year', 'month']].drop_duplicates().sort_values(['year', 'month']).reset_index(drop=True)
        # for i, x in enumerate(unique_months.values):
        #     year = x[0]
        #     month = x[1]
        #     mask = (self.date.year.values == year) & (self.date.month.values == month)

        #     # Build demand and population vector
        #     base_val, pop = self.demand.loc[(self.demand['year'] == year) & (self.demand['month'] == month), ['demand_MG', 'population']].values[0]
        #     self.demand_base[mask] = self.get_daily_demand(base_val, year, month)
        #     self.population[mask] = pop

        #     # Build daily indicator vector
        #     self.indicator[mask] = exogenous_indicator[i]
        
        # create effective demand and active demand copies
        self.demand_eff_base = self.demand_base.copy()
        self.demand_active = self.demand_base.copy()

        # create indicator vector
        self.indicator = exogenous_indicator # should be monthly and the same length as the demand and population vector

        # define curtailment action tracking variables
        self.eff_base_factor = 1.0                                              # Multiplier representing cumulative baseline reductions due to hardening 
        self.event_max_curtail = 0.0                                            # Tracks deepest curtailment order during active drought
        self.current_action = 0                                                 # Tracks current curtailment action index
        self.current_curtailment = 0                                            # Tracks current curtailment rate value
        self.months_in_active_drought = 0                                       # For implementing persistance filter, to be checked against min_hold
        self.consecutive_recovery_months = 0                                    # For reducing action flickering, to be checked against min_recovery
        self.min_hold_condition = False                                         # Tracks minimum hold requirement
        self.min_recovery_condition = False                                     # Tracks minimum recovery period requirement

    def check_drought_exit_conditions(self) -> bool:
        """
        Strategy Checking Function: Evaluates if both persistence filter/minimum hold time
        and minimum recovery time conditions are satisfied to declare drought over.
        """
        hold_period_satisfied = (self.months_in_active_drought >= self.min_hold)
        recovery_period_satisfied = (self.consecutive_recovery_months >= self.min_recovery)
        
        return hold_period_satisfied and recovery_period_satisfied

    def apply_demand_hardening(self):
            """
            Curtailment Order Function: Permanently ratchets down the baseline demand ceiling
            and resets event tracking variables upon drought exit.
            """
            hardened_loss_fraction = self.event_max_curtail * self.f_hardening
            self.eff_base_factor *= (1.0 - hardened_loss_fraction)
            
            # Reset event state counters
            self.event_max_curtail = 0.0
            self.current_action = 0
            self.current_curtailment = 0
            self.months_in_active_drought = 0
            self.consecutive_recovery_months = 0

    def update_monthly_curtailment_policy(self, indicator_thresholds, indicator_val: float):
        """
        Orchestrator Function: Executes indicator lookup, strategy evaluation, 
        and state updates on the 1st day of every month.
        """
        # 1. Lookup raw policy action state from indicator
        raw_action = curtailment_policy(indicator_thresholds, indicator_val)
        
        # 2. Advance monthly counters
        if indicator_val >= indicator_thresholds[0]:  # >= K1
            self.consecutive_recovery_months += 1
        else:
            self.consecutive_recovery_months = 0  # Reset recovery clock if indicator dips
            
        if self.current_action > 0 or raw_action > 0:
            # Increment total time spent under active drought management
            self.months_in_active_drought += 1

        # 3. Evaluate Policy State Transitions
        if self.current_action > 0:
            # --- ACTIVE DROUGHT MANDATE ---
            if raw_action == 0:
                # Indicator suggests normal conditions: Check if strategies permit declaring drought over
                if self.check_drought_exit_conditions():
                    self.apply_demand_hardening()
                else:
                    # Hold current restriction in place until both strategies pass
                    pass
                    
            elif raw_action > self.current_action:
                # ESCALATION: Drought worsens -> Escalate immediately
                self.current_action = raw_action
                
            elif 0 < raw_action < self.current_action:
                # DE-ESCALATION (e.g., Level 3 to Level 1): Requires Strategy 1 hold time
                # if self.months_in_active_drought >= self.min_hold:
                #     self.current_action = raw_action
                self.current_action = raw_action

        else:
            # --- NORMAL OPERATIONS (State 0) ---
            if raw_action > 0:
                # NEW DROUGHT EVENT TRIGGERED
                self.current_action = raw_action
                self.months_in_active_drought = 1
                self.consecutive_recovery_months = 0

        # 4. Update active curtailment rate
        self.current_curtailment = self.curtailment_rates[self.current_action]
        if self.current_action > 0:
            self.event_max_curtail = max(self.event_max_curtail, self.current_curtailment)

    def get_days_in_month(self, year: int, month: int) -> int:
        """
        Natively determines the number of days in a specific month,
        accounting for leap years via inline modulo math.
        """
        # Native inline leap year evaluation
        is_leap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
        
        # Override February to 29 days during a leap year
        if month == 2 and is_leap:
            return 29

        return self.days_in_months_base[month]
      
    def check_and_reset_water_rights(self, month: int, day: int):
        """
        Executes at the start of every single simulated day to clear 
        the cumulative counters on their exact calendar permit anniversaries.
        """
        # September 1st Reset Trigger
        if month == 9 and day == 1:
            self.WR_felton_accum = 0.0
            self.WR_store_accum = 0.0
            self.WR_release_accum = 0.0
            
        # October 1st Reset Trigger
        if month == 10 and day == 1:
            self.WR_b12_accum = 0.0
            self.WR_oakwell_accum = 0.0

    def update_felton_pump_status(self, month: int, day: int):
        """
        Evaluates operational season rules for the Felton Diversion pump.
        Turns off the pump on Jan 1st if Loch Lomond reservoir storage exceeds the safety threshold.
        Automatically enables the pump on Sept 1st for the upcoming water year.
        """
        
        # 1. Compute total current reservoir storage volume across all water accounts
        total_storage = self.V_newell + self.V_felton + self.V_precip
        
        # 2. January 1st Rule Evaluation
        if month == 1 and day == 1:
            if total_storage > self.LL_threshold:
                self.is_felton_pump_on = False
                # print(f"[-] Jan 1st: Loch Lomond storage ({total_storage:.2f} MG) exceeds threshold ({self.LL_threshold:.2f} MG). Felton pump deactivated for the season.")
            else:
                self.is_felton_pump_on = True
                # print(f"[-] Jan 1st: Loch Lomond storage ({total_storage:.2f} MG) below threshold. Felton pump remains ACTIVE.")
                
        # 3. September 1st Reactivation Rule
        if month == 9 and day == 1:
            self.is_felton_pump_on = True
            # print("[-] Sept 1st: Felton pump automatically reactivated for the upcoming season.")
    
    def check_critically_dry(self, year: int, month: int, day: int, lookup_dict: dict):
        """
        Checks if year will be critically dry based on total inflow at bigtrees.
        Updates the critically dry status parameters.
        Only checks once per year because status is fixed for the year.
        """

        # Check if start of new water year
        if month == 10 and day == 1:
            if lookup_dict[year+1] == 1: # year+1 because water year, not calendar year
                self.is_critically_dry_year = True
            else:
                self.is_critically_dry_year = False

    def get_critically_dry(self):
        """
        Returns dictionary mapping water year to 1 if water year is critically dry, 0 otherwise.
        """
        # grab relevant data from flows dataframe
        flow_df = self.flow[['bigtrees']].copy()
        flow_df['year'] = self.date[['year']].values.copy()
        flow_df['month'] = self.date[['month']].values.copy()

        # add water year column
        flow_df['wy_toggle'] = 0
        flow_df.loc[flow_df['month'] >= 10, 'wy_toggle'] = 1
        flow_df['wy'] = flow_df['year'] + flow_df['wy_toggle']

        # drop unwanted columns
        flow_df.drop(columns=['year', 'month', 'wy_toggle'], inplace=True)

        # determine annual inflow based on water year
        ann_flow = flow_df.groupby(by=['wy']).sum().reset_index(drop=False)

        # check if this water year will be critically dry
        ann_flow['crit_dry'] = 0
        ann_flow.loc[ann_flow['bigtrees'] < self.crit_dry_threshold, 'crit_dry'] = 1

        # convert to dictionary for easy look-up
        ann_flow.drop(columns=['bigtrees'], inplace=True)
        key_col = 'wy'
        val_col = 'crit_dry'
        
        return dict(zip(ann_flow[key_col], ann_flow[val_col]))

    def check_first_flush(self, month: int, day: int, bt_flow: float):
        """
        From UMass Model report:
        - Felton water cannot be diverted if the "first flush has not occurred yet
        - first flush represents conditions required to inflate the Felton Dam and enable pumping at Felton Diversion
        - condition: SLR flow at big trees must exceed first flush flow (100 CFS) for ANY 2 days following Sept. 1st
        - assume that the first flush conditions lasts until Sept. 1st of the next year 
        """
        # reset first flush status
        if (month == 9) and (day == 1):
            self.first_flush = False
            self.first_flush_counter = 0

        if not self.first_flush:
            if bt_flow >= (self.first_flush_flow * self.CFS_to_MGD):
                self.first_flush_counter += 1
            if self.first_flush_counter > 1:
                self.first_flush = True
    
    def get_turbidity(self):
        """
        Stores daily turbidity flag vectors for multiple water supply streams. 
        Based on these rules from the UMass Model Report: 
        - Tait and Felton diversions are shut down on the current day plus 2 additional days 
          if current day rainfall exceeds 0.67 inches
        - Laguna and Liddell diversions are shut down on the current day plus 1 additional day
          if current day rainfall exceeds 0.67 inches
        - Majors diversion is shut down on the current day plus the truncated integer value of 
          3 times the current day rainfall additional days if current day rainfall exceeds 0.67 inches
            - For instance, if the daily rainfall is 3.4 inches, the truncated value of 3x3.4=10.2 is 10. 
            - Then, the diversion will be shut down for 11 days.
        
        Assumptions
        - can use average precipitaion for all turbidity flags (instead of watershed specific precipitation)
        """
        # get precipitation data
        precip = np.array(self.weather['precip_mm'].values)
        event = (precip > (self.turb_flag * self.in_to_mm)).astype(np.uint8)

        # SLR turbidity flag
        SLR_flag = event.copy()
        SLR_flag[1:] += event[:-1]
        SLR_flag[2:] += event[:-2]
        self.slr_turb = 1 - SLR_flag
        
        # Liddel and Laguna turbidity flag
        NC_flag = event.copy()
        NC_flag[1:] += event[:-1]
        self.nc_turb = 1 - NC_flag

        # Majors turbidity flag
        MAJ_flag = np.zeros(len(precip), dtype=bool)
        event_days = np.flatnonzero(precip > (self.turb_flag * self.in_to_mm))
        durations = (3 * precip[event_days]).astype(np.int32)
        for i, d in zip(event_days, durations):
            MAJ_flag[i:min(i + d + 1, len(MAJ_flag))] = True
        MAJ_flag = MAJ_flag.astype(np.uint8)
        self.maj_turb = 1 - MAJ_flag

    def get_daily_demand(self, total_monthly_demand: float, year: int, month: int) -> float:
        """
        Dynamically disaggregates demand from monthly to daily
        using the dedicated days-in-month lookup function.
        """
        # Get exact number of days in this month
        days_in_month = self.get_days_in_month(year, month)
        
        # Distribute monthly total evenly over the days of the month
        return total_monthly_demand / days_in_month

    def get_NC_supply(self, year: int, month: int, row: pd.Series) -> tuple:
        """
        Computes North Coast Pipeline supply.
        """
        
        # Extract daily creek inflows from the row
        i_lid = float(row.get('liddel', 0.0))
        i_lag = float(row.get('laguna', 0.0))
        i_maj = float(row.get('majors', 0.0))

        # Extract turbidity flags
        index = int(row.get('t', 0))
        t_nc = self.nc_turb[index]
        t_maj = self.maj_turb[index]
        
        # Determine NC farmer monthly demand using monthly profile and total annual demand
        monthly_fraction = float(self.p_NC_farm.get(month, 0.0)) / 100 # Look up the monthly allocation fraction from profiles
        farmer_monthly_demand = monthly_fraction * self.d_NC_Farm

        # Covert monthly demand to daily demand
        farmer_demand = self.get_daily_demand(farmer_monthly_demand, year, month)

        # water routing logic
        s_lid = min(self.c_lid, i_lid * t_nc)
        s_lag = min(self.c_lag, i_lag * t_nc)
        s_maj = min(self.c_maj, i_maj * t_maj)

        available_creek_flow = s_lid + s_lag + s_maj
        total_NC_diversion = min(available_creek_flow, self.c_NC)
        
        s_NC = max(0.0, total_NC_diversion - farmer_demand)
        unmet_farmer_demand = max(0.0, farmer_demand - total_NC_diversion)
        # met_farmer_demand = farmer_demand - unmet_farmer_demand
        
        return s_NC, unmet_farmer_demand
    
    def get_b12_supply(self, month: int) -> float:
        """
        Computes the daily extraction volume from the Beltz 12 Well.
        Strictly applies the critical dry year restriction, monthly profile capacities, 
        and tracks the running annual water right ledger balance.
        """
        
        # 1. Fetch monthly extraction capacity for this month from the profile dictionary
        c_b12_daily = float(self.c_b12.get(month, 0.0))
        
        # 2. Only pump if a critically dry year has been declared
        if c_b12_daily > 0.0 and self.is_critically_dry_year:
            # 3. Assess remaining permit allowance using the accumulation approach
            remaining_right = max(0.0, self.WR_b12_limit - self.WR_b12_accum)
            
            # 4. determine "today's" supply from Beltz 12 Well
            s_b12 = min(c_b12_daily, remaining_right)
            
            # 5. Accumulate extraction to the water right
            self.WR_b12_accum += s_b12
            return s_b12
            
        return 0.0

    def get_oakwell_supply(self, month: int) -> float:
        """
        Computes the total daily extraction volume from the Live Oak Wells (Beltz Wells 8, 9, 10).
        Applies monthly profile capacities and tracks the running annual water right balance.
        """
        
        # 1. Fetch monthly extraction capacity for this month from the profile dictionary
        c_oakwell_daily = float(self.c_oakwell.get(month, 0.0))
        
        if c_oakwell_daily > 0.0:
            # 2. Assess remaining water right allowance
            remaining_right = max(0.0, self.WR_oakwell_limit - self.WR_oakwell_accum)
            
            # 3. determine "today's" supply from the Live Oak Wells
            s_oakwell = min(c_oakwell_daily, remaining_right)
            
            # 4. Accumulate extraction to the water right
            self.WR_oakwell_accum += s_oakwell
            return s_oakwell
            
        return 0.0
    
    def get_tait_supply(self, row: pd.Series, i_slr_felton: float, month: int) -> float:
        """
        Computes the allowable SLR extraction at the Tait Street Diversion based on i_slr_felton, which 
        is a guess or calculation of the flow downstream of the Felton Diversion in the SLR.
        """

        # Get SLR inflow at Tait Street Diversion (SLR basin between Felton and Tait Diversion)
        i_slr_tait = float(row.get('tait', 0.0))

        # Extract turbidity flags
        index = int(row.get('t', 0))
        t_slr = self.slr_turb[index]
        
        # Look up Tait Well max extraction from monthly profile dictionary
        c_tait_well_daily = float(self.c_tait_well.get(month, 0.0))
            
        # Determine maximum extraction from SLR at Tait Street
        s_tait = max(0.0, min(i_slr_tait * t_slr + i_slr_felton * t_slr + c_tait_well_daily - self.slr_env_buffer, self.c_tait))

        # compute the total tait supply and actual diversion amount for book-keeping
        s_tait_div = max(0.0, s_tait - c_tait_well_daily)

        return s_tait, s_tait_div
    
    def get_hydraulic_constraint(self, reservoir_volume: float) -> float:
        """
        Computes a hydraulic constraint factor based on the current reservoir volume.
        The greater the reservoir volume, the harder the pump has to work.
        """

        if reservoir_volume > 1487:
            # equation obtained from original model from UMass team
            hydraulic_factor = -0.000234 * reservoir_volume + 1.1578  
        else:
            hydraulic_factor = 1

        return hydraulic_factor


    def calculate_discrete_pump_flow(self, felton_supply: float) -> float:
        """
        Takes a continuous available water volume and rounds it down to 
        the nearest discrete pump band capacity.
        """
        # If the water available doesn't even hit the lowest pump speed, pump is off
        if felton_supply < self.felton_pump_rates[0]:
            return 0.0
            
        # Filter pump rates that are less than or equal to the available supply
        allowable_rates = [rate for rate in self.felton_pump_rates if rate <= felton_supply]
        
        return max(allowable_rates) if allowable_rates else 0.0

    def route_san_lorenzo_river(self, date_row: pd.Series, row: pd.Series, s_NC: float, s_b12: float, s_oakwell: float, 
                                city_demand: float, LL_space: float, hyd_const: float) -> tuple:
        """
        Solves the circular dependency between upstream Felton Diversion pumping and downstream Tait Street Diversion 
        using a convergence loop. Updates the persistent Felton extraction water right ledger post-convergence.
        Inputs:
            row             contains flow data
            s_NC            North Coast pipeline supply
            s_b12           Beltz 12 Well supply
            s_oakwell       Live Oak Wells supply
            city_demand     City urban demand for the day
            LL_space        Available space in Loch Lomond reservoir
            hyd_const       Hydraulic constraint for Felton pump, unitless
        
        Output (all outputs are consistent with mass balance at SLR Felton and SLR Tait nodes):
            actual_felton   Felton Diversion flow to LL reservoir
            s_tait          Total supply provided by all sources except Felton and LL Reservoir
            s_tait_div      Tait Street diversion flow
            gap             unmet urban demand after non-Felton and LL reservoir supplies are applied
            i_slr_felton    SLR flow downstream of Felton diversion
        
        non-Felton and LL reservoir supplies = North Coast pipeline, Beltz 12 and Live Oakwells, 
        Tait Street basin inflow, Tait well
        All flows are in MGD, all volumes are in MG
        """

        month = int(date_row.get('month'))
        i_slr_bigtrees = float(row.get('bigtrees', 0.0))

        # Extract turbidity flags
        index = int(date_row.get('t', 0))
        t_slr = self.slr_turb[index]
        
        # Look up dynamic monthly pipeline capacity constraint for Felton Diversion from monthly profiles
        c_felton_daily = float(self.c_felton.get(month, 0.0))
        
        # Convergence parameters
        guess_felton = 0.0      # initial stable guess
        tolerance = 0.01        # relatively relaxed tolerance
        max_iterations = 100    # safety feature to make sure simulation doesn't spin out of control
        iteration_counter = 0
        
        # Placeholders for stable daily variables
        actual_felton = 0.0
        s_tait = 0.0
        gap = 0.0
        s_down_felton = 0.0

        # drought reserve condition
        v_tot = self.V_newell + self.V_felton + self.V_precip

        if self.is_felton_pump_on and (self.first_flush) and (t_slr == 1) and (v_tot < self.LL_reserve):
            # felton pump cannot be used so we know the actual felton diversion
            actual_felton = 0.0
            
            # 1. Flow balance downriver of Felton
            i_slr_felton_actual = max(0.0, i_slr_bigtrees - actual_felton)

            # 2. Compute Tait Street diversion
            s_tait, s_tait_div = self.get_tait_supply(row, i_slr_felton_actual, month)

            # 3. Sum up total downstream municipal supply and calculate current urban demand gap
            s_tait_total = min(s_NC + s_tait, self.c_tait_total)
            s_down_felton = s_tait_total + s_b12 + s_oakwell
            gap = max(0.0, city_demand - s_down_felton)

        else:
            # Need to determine felton diversion using a recursive process to make sure 
            # we don't violate the SLR flow balance (extraction at felton should influence flow at tait)  
            while True:
                # 1. Flow balance downriver of Felton based on current iteration's guess
                i_slr_felton_guess = max(0.0, i_slr_bigtrees - guess_felton)
                
                # 2. Compute Tait Street diversion using the guessed flow
                s_tait, s_tait_div = self.get_tait_supply(row, i_slr_felton_guess, month)
                
                # 3. Sum up total downstream municipal supply and calculate current urban demand gap
                s_tait_total = min(s_NC + s_tait, self.c_tait_total) # enforce flow capacity  
                
                # what I think it should be 
                s_down_felton = s_tait_total + s_b12 + s_oakwell
                gap = max(0.0, city_demand - s_down_felton)

                # based on "leftover_demand" logic from the base model 
                s_down_felton = s_tait_total # doesn't take max total tait flow capacity into account 
                gap = max(0.0, city_demand - s_down_felton)
                
                # 4. Process Felton pump criteria based on the remaining urban shortfall
                if (gap > 0.0) and (c_felton_daily > 0.0):

                    # Flow available at Felton after environmental and municipal protection buffers
                    felton_supply_total = max(0.0, i_slr_bigtrees - self.slr_env_buffer - s_down_felton)
                    
                    # Apply hydraulic capability adjustments
                    felton_pump_limit = min(felton_supply_total, hyd_const * c_felton_daily)
                    
                    # Constrain by available physical storage space in Loch Lomond
                    felton_supply = min(felton_pump_limit, LL_space)
                    
                    # Snap the continuous volume down to a discrete pump step setting
                    felton_pump_flow = self.calculate_discrete_pump_flow(felton_supply)
                else:
                    felton_pump_flow = 0.0
                    
                # 5. Check remaining annual legal extraction allowances
                remaining_felton_right = max(0.0, self.WR_felton_limit - self.WR_felton_accum)
                actual_felton = min(felton_pump_flow, remaining_felton_right)
                
                # 6. Evaluate if the iteration guess has stabilized
                if abs(guess_felton - actual_felton) < tolerance:
                    break
                    
                iteration_counter += 1
                if iteration_counter >= max_iterations:
                    print(f"Convergence warning on {int(date_row['year'])}-{int(date_row['month'])}-{int(date_row['day'])}: Loop capped out at {max_iterations} iterations.")
                    break
                    
                # Update the guess vector for the next iterative sweep
                guess_felton = actual_felton

        # --- Post-Convergence Object State Commitment ---
        # Commit the validated daily pumping volume to our permanent ledger exactly once
        self.WR_felton_accum += actual_felton
        
        # Compute exact remaining river discharge leaving Felton boundary limits
        i_slr_felton = i_slr_bigtrees - actual_felton
        
        return actual_felton, s_tait, s_tait_div, gap, i_slr_felton
    
    def get_reservoir_surface_area(self, total_volume: float) -> float:
        """
        Interpolates the reservoir surface area based on the current 
        total stored volume using bathymetry data.
        Inputs:
            total_volume in MG
        Outputs:
            surface area in acres 
        """
        # Assumes self.bathymetry is a pandas DataFrame with 'volume' and 'area' columns
        if self.bathymetry is None or self.bathymetry.empty:
            return 0.0
        return float(np.interp(total_volume, self.bathymetry['storage_MG'], self.bathymetry['area_acre']))
    
    def get_LL_env_flow(self, month: int, total_volume: float, i_newell: float) -> float:
        """
        Determines mandatory environmental release requirements from Loch Lomond reservoir
        based on current month, total storage volume, and Newell Creek inflow.
        Converts CFS to MGD.
        """
        # Extract Storage Threshold, Low Flow CFS, and High Flow CFS from input data
        threshold, low_cfs, high_cfs = self.env_flows[month]

        # Deterimine required environmental flow and convert to MGD
        selected_cfs = high_cfs if total_volume >= threshold else low_cfs
        env_flow_base = selected_cfs * self.CFS_to_MGD
        
        # July and August special appropriation rules: stream fully appropriated
        # cannot store any Newell Creek water, all input must be spilled 
        if month in [7, 8]:
            return max(env_flow_base, i_newell)
            
        return env_flow_base

    def execute_reservoir_balance(self, month: int, day: int, i_newell: float, i_precip: float, o_evap: float, 
                                  total_o_env:float, actual_o_env: float, actual_d_slv: float,
                                  V_newell_morning: float, V_felton_morning: float, V_precip_morning: float,
                                  s_felton_div: float, urban_demand_gap: float) -> dict:
        """
        Executes the "evening" bookkeeping pass for the Loch Lomond reservoir. 
        Incorporate "afternoon" pumping (water from Felton Diversion), 
        releases as much as possible to satisfy municipal gap (there is no operating policy!), 
        checks for physical spills, and tracks water rights ledger accumulations.
        """
        # --- A: Inject the afternoon Felton pumping water into the morning volume ---
        V_newell_next = V_newell_morning
        V_felton_next = V_felton_morning + s_felton_div
        V_precip_next = V_precip_morning
        
        # --- B: Allocate Priority-Ordered Releases to satisfy the Urban Gap ---
        V_total_pre_release = V_newell_next + V_felton_next + V_precip_next
        
        # Physical maximum extraction capability from the reservoir today
        # Can only transfer to GHWTP if volume if above drought reserve
        if V_total_pre_release > self.LL_reserve:
            release_max = max(0.0, min(self.c_LL_release, V_total_pre_release - self.LL_min))
            release_target = min(urban_demand_gap, release_max)
        else:
            release_target = 0.0
        
        # 1. First priority release: Felton water bucket
        release_felton = min(release_target, V_felton_next)
        V_felton_next -= release_felton
        release_target -= release_felton
        
        # 2. Second priority release: Newell Creek water bucket (constrained by remaining water rights)
        remaining_release_right = max(0.0, self.WR_release_limit - self.WR_release_accum)
        release_newell = min(release_target, V_newell_next, remaining_release_right)
        V_newell_next -= release_newell
        release_target -= release_newell
        self.WR_release_accum += release_newell  # Increment persistent annual release permit tracker
        
        # 3. Third priority release: Direct Precipitation water bucket
        release_precip = min(release_target, V_precip_next)
        V_precip_next -= release_precip
        release_target -= release_precip
        
        # Final residual deficit is water the city needed but the reservoir couldn't provide
        total_reservoir_release = release_felton + release_newell + release_precip
        unmet_urban_demand = max(0.0, urban_demand_gap - total_reservoir_release)

        # --- C: Meet Leftover Environmental Flow and SLV Demand If Possible ---
        
        # First try to meet environmental spills
        # remaining environmental flow water to be spilled
        remaining_env = total_o_env - actual_o_env # this will be zero most of the time

        # Spill from Felton bucket, based on availablity
        env_spill_felton = min(remaining_env, V_felton_next)
        V_felton_next -= env_spill_felton
        remaining_env -= env_spill_felton

        # Spill from precip bucket, based on availablity
        env_spill_precip = min(remaining_env, V_precip_next)
        V_precip_next -= env_spill_precip
        remaining_env -= env_spill_precip # remaining_env is unmet environmental flow

        # Second try to meet environmental spills
        # remaining environmental flow water to be spilled
        remaining_slv = self.d_slv - actual_d_slv

        # Spill from Felton bucket, based on availablity
        slv_felton = min(remaining_slv, V_felton_next)
        V_felton_next -= slv_felton
        remaining_slv -= slv_felton

        # Spill from precip bucket, based on availablity
        slv_precip = min(remaining_slv, V_precip_next)
        V_precip_next -= slv_precip
        remaining_slv -= slv_precip # remaining_slv is unmet SLV demand
        
        # --- D: Physical Overfill Spill Check ---
        V_total_pre_spill = V_newell_next + V_felton_next + V_precip_next
        ll_spill = max(0.0, V_total_pre_spill - self.LL_max)

        # Track remaining spill obligation
        remaining_spill = ll_spill

        # Spill first from Newell bucket, capped by what is actually inside it
        spill_from_newell = min(remaining_spill, V_newell_next)
        V_newell_next -= spill_from_newell
        remaining_spill -= spill_from_newell

        # Spill from Felton bucket next if there's still excess water spilling over
        spill_from_felton = min(remaining_spill, V_felton_next)
        V_felton_next -= spill_from_felton
        remaining_spill -= spill_from_felton

        # Finally, spill from precipitation bucket if needed
        spill_from_precip = min(remaining_spill, V_precip_next)
        V_precip_next -= spill_from_precip
        
        # --- E: End-of-Day Water Rights Accounting ---
        # Evaluate how much net new Newell water was legally captured to charge against the annual storage right
        total_pre_evap = (self.V_newell + i_newell) + self.V_felton + (self.V_precip + i_precip) # Total system volume benchmark
        evap_newell_basis = 0.0
        if total_pre_evap > 0.0:
            # Estimate the proportional amount of evaporation that hit the Newell bucket
            evap_newell_basis = ((self.V_newell + i_newell) / total_pre_evap) * o_evap
            
        delta_v_newell_basis = max(0.0, i_newell - evap_newell_basis - actual_o_env - actual_d_slv - spill_from_newell)
        remaining_store_right = max(0.0, self.WR_store_limit - self.WR_store_accum)
        excess_newell_storage = max(0.0, delta_v_newell_basis - remaining_store_right)
        
        # If we exceeded our legal right to store this water, it is forced to spill paper-wise
        # Ensure paper rights corrections can never drag the physical bucket below zero
        actual_paper_spill = min(excess_newell_storage, V_newell_next)
        ll_spill += actual_paper_spill
        V_newell_next -= actual_paper_spill
        self.WR_store_accum += max(0.0, delta_v_newell_basis - actual_paper_spill)
        
        # --- F. Commit Finalized Values back to persistent Class Memory variables ---
        self.V_newell = V_newell_next
        self.V_felton = V_felton_next
        self.V_precip = V_precip_next
        
        # --- G: End-of-Year Accounting (September 1st Pool Consolidation) ---
        if month == 9 and day == 1:
            self.V_newell = self.V_newell + self.V_felton + self.V_precip
            self.V_felton = 0.0
            self.V_precip = 0.0
            
        return total_reservoir_release, ll_spill, unmet_urban_demand, remaining_env, remaining_slv
    
    def run_simulation(self, indicator_thresholds, results="objective"):
        """
        Prepares data frameworks and orchestrates the daily multi-year water systems balance simulation loop.

        'results' input controls level of detail of results the simulation returns 
            "operations"    returns the operations-relevant results (supply sources, flow routing, reservoir storage, etc.).
            "demand"        results are demand-relevant (i.e., effective baseline and active demand, curtailment actions, etc.)
            "objective"     returns performance objective value (float)
        By default, it is set to "objective"

        Output:

        results_df: systems model simulation results dataframe (results = "full" or "minimal")
        metric: float = weight_1 * total urban demand deficit + weight_2 * total curtailed demand + HR2W penalty * total demand below HR2W
        
        Notes:
        Maintains proportional evaporation and priority-ordered municipal releases.
        Keeps track of three different reservoir volumes:
            1. Newell Creek water
            2. Felton Diversion Water
            3. Direct Precipitation Water
        
        Order of priority when releasing water to GHWTP:
            1. Felton water
            2. Newell water
            3. precipitation water
        
        It is assumed that only Newell water will be used for the following releases:
            1. Release to meet San Lorenzo Valley daily demand
            2. Minimum environmental flow releases
            3. Reservoir spill releases
        
        Other assumptions
            1. Reservoir outflows due to evaporation are proportionally removed from all three sources 
            2. If there is not enough Newell Creek water to satisfy environmental flows AND SLV demand,
               environmental flows take priority, then SLV demand.
            
        """
        
        # Determine number of days/timesteps
        num_days = len(self.date)

        # get critically dry status lookup dictionary
        crit_dry_dict = self.get_critically_dry()

        # update daily turbidity flag vectors
        self.get_turbidity()
        
        if results == 'operations':
            # Pre-allocate output arrays to store results
            #   Using pre-allocated numpy arrays completely eliminates inner-loop memory bottlenecks
            outputs = {
                "s_NC": np.zeros(num_days),
                "s_b12": np.zeros(num_days),
                "s_oakwell": np.zeros(num_days),
                "s_tait": np.zeros(num_days),
                "felton_div": np.zeros(num_days),
                "tait_div": np.zeros(num_days),
                "i_slr_felton": np.zeros(num_days),
                "i_slr_ocean": np.zeros(num_days),
                "V_newell": np.zeros(num_days),
                "V_felton": np.zeros(num_days),
                "V_precip": np.zeros(num_days),
                "V_total": np.zeros(num_days),
                "I_precip": np.zeros(num_days),
                "releases": np.zeros(num_days),
                "evap_loss": np.zeros(num_days),
                "env_spill": np.zeros(num_days),
                "other_spill": np.zeros(num_days),
                "urban_demand_gap": np.zeros(num_days),
                "active_urban_demand": np.zeros(num_days),
                "unmet_urban_demand": np.zeros(num_days),
                "unmet_farmer_demand": np.zeros(num_days),
                "unmet_env_flow": np.zeros(num_days),
                "unmet_slv_demand": np.zeros(num_days),
                "mass_balance_check": np.zeros(num_days)
            }
        elif results == "demand":
            outputs = {
                "NC_supply_MGD": np.zeros(num_days),
                "GW_supply_MGD": np.zeros(num_days),
                "tait_div_MGD": np.zeros(num_days),
                "felton_div_MGD": np.zeros(num_days),
                "LL_vol_MG": np.zeros(num_days),
                "LL_release_MGD": np.zeros(num_days),
                "curtailment": np.zeros(num_days),
                "eff_base_factor": np.zeros(num_days),
                "event_max_curtail": np.zeros(num_days),
                "months_in_active_drought": np.zeros(num_days),
                "consecutive_recovery_months": np.zeros(num_days),
                "base_demand_MGD": np.zeros(num_days),
                "eff_base_demand_MGD": np.zeros(num_days),
                "active_demand_MGD": np.zeros(num_days),
                "unmet_urban_demand": np.zeros(num_days),
            }
        else:
            urban_deficit = np.zeros(num_days) # just need this array to track unmet urban demand

        # Run Water Balance Simulation
        #   We process daily timesteps sequentially to respect historical continuity 
        #   and allow state variables to pass cleanly from day t to day t+1.
        
        # print(f"Running Water Balance Engine, processing {num_days} simulation days...")

        # for t in tqdm(range(num_days), desc="Day", ncols=100, colour="green", mininterval=1, maxinterval=5):

        for t in range(num_days):

            # =========================================================================
            # 1. "MORNING" RESERVOIR CALCULATIONS (WEATHER, ENVIRONMENT FLOWS, SLV)
            # =========================================================================
            row = self.flow.iloc[t]
            date_row = self.date.iloc[t]
            year = int(date_row.get('year'))
            month = int(date_row.get('month'))
            day = int(date_row.get('day'))

            # A. Update seasonal/anniversary flags
            self.check_and_reset_water_rights(month, day)
            self.check_first_flush(month, day, float(row.get('bigtrees', 0.0)))
            self.update_felton_pump_status(month, day)
            self.check_critically_dry(year, month, day, crit_dry_dict)
            
            # B. Reservoir Physical Dynamics (Using yesterday's closing storage)
            v_total_start = self.V_newell + self.V_felton + self.V_precip # total volume in MG
            area_res = self.get_reservoir_surface_area(v_total_start) # surface area in acres

            # C. Newell Creek Inflow, Direct Precipitation, and Evaporation 
            i_newell = float(row.get('newell', 0.0))
            precip_rate = float(self.weather.iloc[t].get('precip_mm', 0.0)) # precipitation in mm/day
            i_precip = (precip_rate * (1 / self.ft_to_mm)) * area_res * (self.AF_to_MG) # direct precipitaiton inflow in MGD
            evap_rate = float(self.weather.iloc[t].get('evap_mm', 0.0)) # evaporation in mm/day
            o_evap = (evap_rate * (1 / self.ft_to_mm)) * area_res * (self.AF_to_MG) # reservoir evaporation in MGD

            # D. Determine minimum environmental flow requirements
            o_env = self.get_LL_env_flow(month, v_total_start, i_newell)

            # E. Compute available space in reservoir
            # Proportional evaporation splitting
            total_V_pre_evap = (self.V_newell + i_newell) + (self.V_felton) + (self.V_precip + i_precip)
            evap_newell = ((self.V_newell + i_newell) / total_V_pre_evap) * o_evap if total_V_pre_evap > 0 else 0.0
            evap_felton = ((self.V_felton) / total_V_pre_evap) * o_evap if total_V_pre_evap > 0 else 0.0
            evap_precip = ((self.V_precip + i_precip) / total_V_pre_evap) * o_evap if total_V_pre_evap > 0 else 0.0

            # Dynamic outflow capping of Newell Creek water
            # Track what is physically available in the Newell bucket after natural inflows and evaporation
            v_newell_avail = max(0.0, self.V_newell + i_newell - evap_newell)
            
            # Environmental flows have first priority on Newell water
            actual_o_env = min(o_env, v_newell_avail)
            v_newell_avail -= actual_o_env
            
            # San Lorenzo Valley demand has second priority on Newell water
            actual_d_slv = min(self.d_slv, v_newell_avail)
            v_newell_avail -= actual_d_slv

            # Establish safe, physically guaranteed morning volumes
            V_newell_morning = v_newell_avail
            V_felton_morning = max(0.0, self.V_felton - evap_felton)
            V_precip_morning = max(0.0, self.V_precip + i_precip - evap_precip)

            # "Morning" reservoir volume
            V_total_morning = V_newell_morning + V_felton_morning + V_precip_morning
            LL_space = self.LL_max - V_total_morning

            # =========================================================================
            # 2. GATHER DEMAND & NON-FELTON & RESERVOIR SUPPLY INFORMATION
            # =========================================================================
            
            # A. Apply curtailment policy at the monthly timestep
            if day == 1:
                current_month_idx = self.month_to_idx[(year, month)]
                self.update_monthly_curtailment_policy(indicator_thresholds, self.indicator[current_month_idx])
                self.demand_eff_base[current_month_idx] = self.eff_base_factor * self.demand_base[current_month_idx]
                self.demand_active[current_month_idx] = (1 - self.current_curtailment) * self.demand_eff_base[current_month_idx]
            
            # B. Get total urban demand for today (Disaggregating monthly baseline to daily)
            # current_month_idx = self.month_to_idx[(year, month)]
            d_urban_daily = self.get_daily_demand(self.demand_active[current_month_idx], year, month)

            # C. Gather North Coast Pipeline surface supplies and 
            # Beltz 12 Well and Live Oak Wells groundwater supplies
            s_NC, unmet_farm = self.get_NC_supply(year, month, row)
            s_b12 = self.get_b12_supply(month)
            s_oakwell = self.get_oakwell_supply(month)

            # =========================================================================
            # 3. AFTERNOON ROUTING: FELTON & TAIT DIVERSION SLR MASS BALANCE CONVERGENCE 
            # =========================================================================
            
            # A. Determine hydraulic constraint factor based on current reservoir volume
            hyd_const = self.get_hydraulic_constraint(V_total_morning)
            
            # B. Settle the recursive dependency between upstream Felton Diverion and downstream Tait Street Diversion
            slr_routing = self.route_san_lorenzo_river(date_row, row, s_NC, s_b12, s_oakwell, d_urban_daily, LL_space, hyd_const)
            s_felton_div, s_tait, s_tait_div, urban_demand_gap, i_slr_felton = slr_routing

            # =========================================================================
            # 4. "EVENING" BOOKKEEPING: RESERVOIR MASS BALANCE & WATER RIGHTS
            # =========================================================================
            # A. Execute reservoir mass balance and water rights updates
            reservoir_summary = self.execute_reservoir_balance(month, day, i_newell, i_precip, o_evap,
                                                               o_env, actual_o_env, actual_d_slv,
                                                               V_newell_morning, V_felton_morning, V_precip_morning,
                                                               s_felton_div, urban_demand_gap)
            total_reservoir_release, ll_spill, unmet_urban_demand, remaining_env, remaining_slv = reservoir_summary
            
            # B. Compute Mass Balance Check change in volume = delta V = sum of inputs - sum of outputs
            change_in_volume = (self.V_newell + self.V_felton + self.V_precip) - v_total_start
            delta_V = (i_newell + i_precip + s_felton_div) - (total_reservoir_release + o_evap + (o_env - remaining_env) + (self.d_slv - remaining_slv) + ll_spill)

            if results == "operations":
                # Save results to pre-allocated output arrays
                outputs["s_NC"][t] = s_NC
                outputs["s_b12"][t] = s_b12
                outputs["s_oakwell"][t] = s_oakwell
                outputs["s_tait"][t] = s_tait
                outputs["felton_div"][t] = s_felton_div
                outputs["tait_div"][t] = s_tait_div
                outputs["i_slr_felton"][t] = i_slr_felton
                outputs["i_slr_ocean"][t] = i_slr_felton + float(row.get('tait', 0.0)) - s_tait_div
                outputs["V_newell"][t] = self.V_newell
                outputs["V_felton"][t] = self.V_felton
                outputs["V_precip"][t] = self.V_precip
                outputs["V_total"][t] = self.V_newell + self.V_felton + self.V_precip
                outputs["I_precip"][t] = i_precip
                outputs["releases"][t] = total_reservoir_release
                outputs["evap_loss"][t] = o_evap
                outputs["env_spill"][t] = o_env
                outputs["other_spill"][t] = ll_spill
                outputs["urban_demand_gap"][t] = urban_demand_gap
                outputs["active_urban_demand"][t] = d_urban_daily
                outputs["unmet_urban_demand"][t] = unmet_urban_demand
                outputs["unmet_farmer_demand"][t] = unmet_farm
                outputs["unmet_env_flow"][t] = remaining_env
                outputs["unmet_slv_demand"][t] = remaining_slv
                outputs["mass_balance_check"][t] = change_in_volume - delta_V
            elif results == "demand":
                outputs["NC_supply_MGD"][t] = s_NC
                outputs["GW_supply_MGD"][t] = s_b12 + s_oakwell + (s_tait - s_tait_div)
                outputs["tait_div_MGD"][t] = s_tait_div
                outputs["felton_div_MGD"][t] = s_felton_div
                outputs["LL_vol_MG"][t] = self.V_newell + self.V_felton + self.V_precip
                outputs["LL_release_MGD"][t] = total_reservoir_release
                outputs["curtailment"][t] = self.current_curtailment
                outputs["eff_base_factor"][t] = self.eff_base_factor
                outputs["event_max_curtail"][t] = self.event_max_curtail
                outputs["months_in_active_drought"][t] = self.months_in_active_drought
                outputs["consecutive_recovery_months"][t] = self.consecutive_recovery_months
                outputs["base_demand_MGD"][t] = self.get_daily_demand(self.demand_base[current_month_idx], year, month)
                outputs["eff_base_demand_MGD"][t] = self.get_daily_demand(self.demand_eff_base[current_month_idx], year, month)
                outputs["active_demand_MGD"][t] = d_urban_daily
                outputs["unmet_urban_demand"][t] = unmet_urban_demand
            else:
                urban_deficit[t] = unmet_urban_demand

        # =========================================================================
        # 5. POST-SIMULATION CONSOLIDATION & RETURN
        # =========================================================================
        # print("Simulation complete. Consolidating daily time-series matrices...")
        
        if results != "objective":
            # Convert the tracking dictionary of numpy arrays into a clean Pandas DataFrame
            # reuse the original flow time-series index to preserve date context (year, month, day)
            simulation_results_df = pd.DataFrame(outputs, index=self.date.index)
            
            # Attach calendar date context columns from the input data for easy plotting
            simulation_results_df['year'] = self.date.year.values
            simulation_results_df['month'] = self.date.month.values
            simulation_results_df['day'] = self.date.day.values
            
            # print("Success! Water Balance Model execution finished cleanly.")
            
            return simulation_results_df
        else:
            # compute performance metric
            total_deficit = np.sum(urban_deficit)
            total_curt = np.sum(self.demand_base - self.demand_active)
            penalty_per_capita = np.clip((self.HR2W * (365/12)) - ((self.demand_active * 1000000)  / self.population), a_min = 0, a_max=None)
            penalty = np.sum((penalty_per_capita * self.population) / 1000000)
            # return [total_deficit, total_curt, penalty]
            return self.def_w * total_deficit + self.curt_cost_w * total_curt + self.penalty_w * penalty