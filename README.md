
# Long Range Default Probabilities

This project implements various models to calculate long-range default probabilities for companies. The primary models used in this project are Merton, Necula, and Rostek. The project also includes tools for calibration, result combination, and visualization of the calculated probabilities.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Usage](#usage)
- [Models](#models)
  - [Merton](#merton)
  - [Necula](#necula)
  - [Rostek](#rostek)
- [Tools](#tools)
- [Visualizations](#visualizations)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

The Long Range Default Probabilities project provides a comprehensive framework for evaluating the credit risk of companies over different maturities (1Y, 5Y, and 10Y) using advanced quantitative models. The main features of this project include:

- Calibration of credit risk models.
- Calculation of default probabilities, mu, sigma, and Hurst coefficients.
- Visualization of results for easy interpretation.

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone https://github.com/k-bargaoui/LongRangeDefaultProbas.git
cd LongRangeDefaultProbas
pip install -r requirements.txt
```

### Requirements

This project relies on the following Python packages:

- `numpy`
- `pandas`
- `matplotlib`
- `scipy`
- `scikit-learn`

The `requirements.txt` file includes all necessary libraries, and you can install them using:

```bash
pip install -r requirements.txt
```

## Usage

The project is structured into several modules, each dedicated to a specific model or functionality. Below are details on how to use each model and the associated tools.

### Models

#### Merton

The Merton model estimates default probabilities based on the firm's market value and debt levels. You can use it as follows:

```python
from merton import Merton

# Load your market cap and debt data as DataFrames
market_cap_data = ...  # Load your data here
debt_data = ...        # Load your data here

# Initialize and calibrate the Merton model
merton_model = Merton(ticker='PIA IM Equity', market_cap=market_cap_data, debt=debt_data, T=5)
result = merton_model.calibrate()

# Access results
print("Default Probability:", result["default_probability"])
print("Mu:", result["mu"])
print("Sigma:", result["sigma"])
```

#### Necula

The Necula model provides a different methodology for calculating default probabilities. Usage is similar to the Merton model:

```python
from necula import Necula

# Initialize and calibrate the Necula model
necula_model = Necula(ticker='PIA IM Equity', market_cap=market_cap_data, debt=debt_data, T=5)
result = necula_model.calibrate()

# Access results
print("Default Probability:", result["default_probability"])
print("Mu:", result["mu"])
print("Sigma:", result["sigma"])
```

#### Rostek

The Rostek model applies an alternative approach for assessing default probabilities.

```python
from rostek import Rostek

# Initialize and calibrate the Rostek model
rostek_model = Rostek(ticker='PIA IM Equity', market_cap=market_cap_data, debt=debt_data, T=5)
result = rostek_model.calibrate()

# Access results
print("Default Probability:", result["default_probability"])
print("Mu:", result["mu"])
print("Sigma:", result["sigma"])
```

### Tools

The `Tools` class facilitates the calibration of multiple models and the consolidation of results across different tickers and maturities. 

```python
from helpers import Tools

# Load your market cap and debt data as DataFrames
market_cap_data = ...  # Load your data here
debt_data = ...        # Load your data here

# Initialize Tools with your market data and ticker list
tools = Tools(market_cap=market_cap_data, debt=debt_data, ticker_list=['PIA IM Equity', 'CO FP Equity'])
tools.calibrate_and_combine_results(['PIA IM Equity', 'CO FP Equity'])

# Get combined results
results = tools.get_results()
print(results)
```

### Visualizations

The project includes visualization functions to plot the calculated probabilities and other metrics. 

```python
from visualisations import plot_output

# Visualize default probabilities for a specific ticker
plot_output(dataframe=results, value='DP(%)', specific_ticker='PIA IM Equity')

# Visualize Hurst coefficients
tools.plot_hurst_coeffs(specific_ticker='PIA IM Equity')
```

## Examples

### Example: Calibrating and Visualizing Results

Here’s a complete example of how to calibrate the models for multiple tickers and visualize the results:

```python
from common_imports import pd
from helpers import Tools
from visualisations import plot_output

# Load market cap and debt data
market_cap_data = pd.read_excel('Data/Data_issuers.xlsx', sheet_name='Mod Market Cap')
debt_data = pd.read_excel('Data/Data_issuers.xlsx', sheet_name='Gross Debt', nrows=1)

# Initialize Tools and calibrate models
tools = Tools(market_cap=market_cap_data, debt=debt_data, ticker_list=['PIA IM Equity', 'CO FP Equity'])
tools.calibrate_and_combine_results(['PIA IM Equity', 'CO FP Equity'])

# Get and visualize results
results = tools.get_results()
plot_output(dataframe=results, value='DP(%)')
```
## Documentation

An HTML page containing the description of all functions can be found in "lib_doc.html"

## Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please feel free to submit a pull request or open an issue on GitHub.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
