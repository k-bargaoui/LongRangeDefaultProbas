from common_imports import plt, pd

def plot_output(dataframe, value, specific_ticker=None):
    """
    Plot specified values as a function of maturity for given tickers.

    Parameters:
    - dataframe: DataFrame
        The DataFrame containing the data to plot, with 'Ticker', 'Maturity', and 'Model' columns.
    - value: str
        The column name in the DataFrame to plot.
    - specific_ticker: str, optional
        The ticker to filter the DataFrame (default is None, which plots all tickers).
    """
    # Determine which tickers to plot
    tickers = [specific_ticker] if specific_ticker else dataframe['Ticker'].unique()

    for ticker in tickers:
        # Filter DataFrame by ticker
        df_ticker = dataframe[dataframe['Ticker'] == ticker].copy()

        # Ensure Maturity is numeric
        df_ticker['Maturity'] = pd.to_numeric(df_ticker['Maturity'], errors='coerce')

        # Pivot the DataFrame for plotting
        df_pivot = df_ticker.pivot(index='Maturity', columns='Model', values=value)

        # Create the plot
        plt.figure(figsize=(5, 4))

        # Plot each model's values
        for model in df_pivot.columns:
            plt.plot(df_pivot.index, df_pivot[model], label=model)

        # Customize plot aesthetics
        plt.title(f'{value} as a Function of Maturity for {ticker}')
        plt.xlabel('Maturity (years)')
        plt.ylabel(value)
        plt.legend(title='Model')
        plt.grid(True)

        # Show the plot
        plt.show()
