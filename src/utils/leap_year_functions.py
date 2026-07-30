import numpy as np

def get_leap_year_indices(start_year: int, z_years: int) -> list[int]:
    """
    Returns 0-based indices of leap years in a block of `z_years` starting at `start_year`.
    
    Handles standard calendar rules (including century non-leap years like 2100).
    """
    leap_indices = []
    
    for i in range(z_years):
        current_year = start_year + i
        
        # Standard leap year check:
        # Divisible by 4 AND (not divisible by 100 OR divisible by 400)
        is_leap = (current_year % 4 == 0) and (
            current_year % 100 != 0 or current_year % 400 == 0
        )
        
        if is_leap:
            leap_indices.append(i)
            
    return leap_indices

def impute_leap_days(mat: np.ndarray, leap_year_indices: list[int] = None) -> np.ndarray:
    """
    Imputes Feb 29 weather data using a 1-NN approach across chunked sets.

    Parameters
    ----------
    mat : np.ndarray
        Shape (y, z, 365, 3) where:
        y = number of sets
        z = number of years per set
        365 = days per year (0-indexed: Feb 28 = index 58, Mar 1 = index 59)
        3 = variables [Tmax, Tmin, Precip]
    leap_year_indices : list of int, optional
        Zero-based year indices within each set `z` that are leap years (e.g., [3, 7, 11...]).
        If None, defaults to standard 4-year cycle (index 3, 7, 11...).

    Returns
    -------
    imputed_mat : np.ndarray
        Shape (y, z, 366, vars_count) with imputed Feb 29 inserted between Feb 28 and Mar 1.
    """
    y, z, days, vars_count = mat.shape
    assert days == 365, "Input matrix must have 365 days per year."

    # Day indices in non-leap 365-day calendar
    FEB28_IDX = 58
    MAR01_IDX = 59

    # Default leap year pattern: years 3, 7, 11... (every 4th year)
    if leap_year_indices is None:
        leap_year_indices = [yr for yr in range(z) if (yr + 1) % 4 == 0]

    # Pre-allocate output array for 366 days
    imputed_mat = np.zeros((y, z, 366, vars_count), dtype=mat.dtype)

    for set_idx in range(y):
        set_data = mat[set_idx] 

        for yr_idx in range(z):
            # Copy original 365 days up to Feb 28 (0 to 58 inclusive -> 59 days)
            imputed_mat[set_idx, yr_idx, :59] = set_data[yr_idx, :59]
            # Copy Mar 1 to Dec 31 into shifted positions (59 to 364 original -> 60 to 365 output)
            imputed_mat[set_idx, yr_idx, 60:] = set_data[yr_idx, 59:]

            # If it's a leap year, perform 1-NN imputation for index 59 (Feb 29)
            if yr_idx in leap_year_indices:
                # 1. Target vector for this leap year: average of its own Feb 28 and Mar 1 (Tmax, Tmin only)
                target_temp = np.mean(
                    [set_data[yr_idx, FEB28_IDX, :2], set_data[yr_idx, MAR01_IDX, :2]], 
                    axis=0
                )

                # 2. Build candidate pool from ALL OTHER years within this set
                candidate_days = []
                for donor_yr in range(z):
                    if donor_yr == yr_idx:
                        continue  # Exclude self
                    # Add Feb 28 and Mar 1 from donor year
                    candidate_days.append(set_data[donor_yr, FEB28_IDX])
                    candidate_days.append(set_data[donor_yr, MAR01_IDX])
                
                candidate_days = np.array(candidate_days)  # Shape: (2*(z-1), 3)

                # 3. Calculate Euclidean distance on [Tmax, Tmin] (first 2 columns)
                candidate_temps = candidate_days[:, :2]
                distances = np.linalg.norm(candidate_temps - target_temp, axis=1)

                # 4. Pick nearest neighbor (1-NN)
                best_match_idx = np.argmin(distances)
                best_match_day = candidate_days[best_match_idx]  # Full array: [Tmax, Tmin, Precip]

                # 5. Assign match to Feb 29 (index 59)
                imputed_mat[set_idx, yr_idx, 59] = best_match_day
            else:
                # Non-leap years: duplicate Feb 28 into Feb 29 position or leave as nan / placeholder
                # Here we fill with np.nan to explicitly mark non-leap Feb 29s
                imputed_mat[set_idx, yr_idx, 59] = np.nan

    return imputed_mat
