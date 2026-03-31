# import necessary libraries
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import datetime as dt

from scipy import stats
from matplotlib.ticker import FuncFormatter
from matplotlib.lines import Line2D

def simulate_index_paths(index_start_value, accumulation_length, annual_rate_of_return, annual_volatility, num_paths, sim_seed=42):
    # Set up the deterministic index growth path
    months = [0]
    index_values = [index_start_value]
    for each in range(1, accumulation_length*12 + 1):
        months.append(each)
        index_values.append(index_values[-1] * np.exp(annual_rate_of_return/12))

    constant_growth_index = pd.DataFrame({'Months': months, 'Index Value': index_values})

    index_end_value = index_values[-1]
    monthly_volatility = annual_volatility / np.sqrt(12)

    # Simulate random paths
    np.random.seed(sim_seed)  # for reproducibility

    # Store all paths
    all_paths = []
    return_paths = []

    for i in range(num_paths):
        # Generate random monthly returns with same mean but with volatility
        random_returns = np.random.normal(annual_rate_of_return/12, monthly_volatility, len(months)-1)

        # Adjust the returns so the final value matches index_end_value
        unadjusted_end_value = index_start_value * np.exp(np.sum(random_returns))
        adjustment = (np.log(index_end_value) - np.log(unadjusted_end_value)) / len(random_returns)
        adjusted_returns = random_returns + adjustment

        # Generate the random index path
        random_index_values = [index_start_value]
        for ret in adjusted_returns:
            random_index_values.append(random_index_values[-1] * np.exp(ret))

        all_paths.append(random_index_values)
        return_paths.append(adjusted_returns)

    # Convert to DataFrame for easier handling
    paths_df = pd.DataFrame(all_paths).T
    paths_df.index = months
    paths_df.columns = [f'Path_{i+1}' for i in range(num_paths)]

    result = pd.concat([constant_growth_index, paths_df], axis=1)

    return result


def calculate_terminal_values(sim_paths, annual_rate_of_return, monthly_contribution):
    months = sim_paths['Months']
    
    # Calculate terminal value for the steady growth path
    steady_terminal_value = 0
    for i, month in enumerate(months):
        if i < len(months) - 1:  # Don't contribute in the last month (month 360)
            # Calculate how many months this contribution will grow
            months_to_grow = len(months) - 1 - i
            # Calculate the growth factor for this contribution
            growth_factor = np.exp(annual_rate_of_return / 12 * months_to_grow)
            # Add this contribution's terminal value
            steady_terminal_value += monthly_contribution * growth_factor

    # Calculate terminal values for all random paths
    random_terminal_values = []

    for path_col in sim_paths.columns:
        if path_col in ['Months', 'Index Value']:
            continue  # Skip the steady growth path
        terminal_value = 0

        for i in range(len(months) - 1):  # Don't contribute in the last month
            # Get the index value at contribution time and at the end
            index_at_contribution = sim_paths.loc[i, path_col]
            index_at_end = sim_paths.loc[months.iloc[-1], path_col]

            # Calculate shares purchased with this contribution
            if index_at_contribution == 0:
                shares_purchased = 0
            else:
                shares_purchased = monthly_contribution / index_at_contribution

            # Calculate value of these shares at the end
            terminal_value += shares_purchased * index_at_end

        random_terminal_values.append(terminal_value)

    # Create summary DataFrame
    terminal_values_summary = pd.DataFrame({
        'Path': ['Steady Growth'] + list(sim_paths.columns[2:]),
        'Terminal Value': [steady_terminal_value] + random_terminal_values
    })
    return terminal_values_summary