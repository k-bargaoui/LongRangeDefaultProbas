from common_imports import pd

def load_data(file_path, start_date, end_date):
    #load market cap
    market_cap = pd.read_excel(file_path, sheet_name="Mod Market Cap")
    market_cap = market_cap.set_index("Dates").loc[start_date:end_date]
    # Load debt data (first row only)
    debt = pd.read_excel(file_path, sheet_name="Gross Debt", nrows=1)
    return market_cap, debt