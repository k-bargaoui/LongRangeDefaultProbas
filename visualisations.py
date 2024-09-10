from common_imports import plt, pd

def plot_probability(dataframe, ticker):
    plt.figure()

    # Ensure 'Maturity' is treated as a numeric column
    dataframe['Maturity'] = pd.to_numeric(dataframe['Maturity'], errors='coerce')

    # Pivot the DataFrame
    df_pivot = dataframe.pivot(index='Maturity', columns='Model', values='default proba')

    # Plot the data
    plt.figure()
    df_pivot.plot(ax=plt.gca())  # Plot using the pivoted DataFrame
    plt.legend()
    plt.title(f"Default probability vs. maturity T for {ticker}")
    plt.xlabel("Maturity (T)")
    plt.ylabel("Probability")
    plt.show()
