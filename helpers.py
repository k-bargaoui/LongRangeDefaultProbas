import pandas as pd

def load_all_data(file_path="Data/Data_issuers.xlsx"):
    """
    Load market capitalization and debt data from an Excel file.
    
    Parameters:
    - file_path: str, optional
        The path to the Excel file (default is "Data/Data_issuers.xlsx").
        
    Returns:
    - market_cap_data: DataFrame
        Market capitalization data.
    - debt: DataFrame
        Company debt information.
    """
    market_cap_data = pd.read_excel(file_path, sheet_name="Mod Market Cap")
    debt = pd.read_excel(file_path, sheet_name="Gross Debt", nrows=1)
    return market_cap_data, debt

def load_window(market_cap_data, start_date, end_date):
    """
    Filter market capitalization data for a specific date range.
    
    Parameters:
    - market_cap_data: DataFrame
        The DataFrame containing market capitalization data with a "Dates" column.
    - start_date: str
        The start date for filtering (format: 'YYYY-MM-DD').
    - end_date: str
        The end date for filtering (format: 'YYYY-MM-DD').
        
    Returns:
    - DataFrame
        Filtered market capitalization data for the specified date range.
    """
    return market_cap_data.set_index("Dates").loc[start_date:end_date]
