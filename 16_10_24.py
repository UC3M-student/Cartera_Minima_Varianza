##### IT WORKS PERFECTLY #####

import yfinance as yf
import pandas as pd
import numpy as np
import cvxpy as cp

def get_yahoo_data_table_format(
    tickers_list,
    benchmark_ticker,
    start_date,
    end_date,
    interval,
    tickers_list_data_to_excel_yes_or_no,
    benchmark_data_to_excel_yes_or_no
):
    """
    Descarga datos de Yahoo Finance, calcula rendimientos y optimiza un portafolio
    minimizando el tracking error respecto a un benchmark.

    Parameters:
    tickers_list: List of tickers to download
    benchmark_ticker: Benchmark ticker to compare with
    start_date: Start date for the data
    end_date: End date for the data
    interval: Data interval (e.g., daily, weekly)
    tickers_list_data_to_excel_yes_or_no: Flag to save tickers data to Excel
    benchmark_data_to_excel_yes_or_no: Flag to save benchmark data to Excel
    """

    # Descargar datos de los activos
    data = yf.download(tickers_list, start=start_date, end=end_date, interval=interval)['Close']

    # Eliminar zonas horarias y columnas con valores NaN
    data.index = data.index.tz_localize(None)
    data = data.dropna(axis=1)

    if data.empty:
      raise ValueError("Downloaded data is empty after filtering NaN columns.")

    # Guardar los datos en un archivo Excel si se requiere
    if tickers_list_data_to_excel_yes_or_no.lower() == "yes":
        data.to_excel("data.xlsx")

    # Convertir los precios en una matriz de retornos
    data_matrix_returns = matrix_simple_returns(data)

    # Descargar datos del benchmark
    benchmark_data = yf.download(benchmark_ticker, start=start_date, end=end_date, interval=interval)['Close']

    if benchmark_data.empty:
      raise ValueError("Downloaded data is empty after filtering NaN columns.")

    # Convertir los precios en una matriz de retornos
    benchmark_data_matrix_return = matrix_simple_returns(benchmark_data)

    # Guardar los datos del benchmark en un archivo Excel si se requiere
    if benchmark_data_to_excel_yes_or_no.lower() == "yes":
        benchmark_data.to_excel("benchmark_data.xlsx")

    # Asegurarse de que las dimensiones de las matrices coincidan
    n_length_benchmark_data = benchmark_data_matrix_return.shape[0]
    data_matrix_returns = data_matrix_returns[-n_length_benchmark_data:]

    # Calcular el exceso de retorno
    excess_return = benchmark_data_matrix_return[:, np.newaxis] - data_matrix_returns

    # Calcular la matriz de covarianza del exceso de retornos
    excess_ret_sigma = np.cov(excess_return, rowvar=False)

    # Optimización para minimizar el tracking error
    number_of_assets = data_matrix_returns.shape[1]
    optimal_weights, min_tracking_error = optimize_portfolio(excess_ret_sigma, number_of_assets)

    # Mostrar resultados
    print("Ponderaciones óptimas:", optimal_weights)
    print("Tracking Error Mínimo:", min_tracking_error)

    return data_matrix_returns


def data_converter_in_matrix_return(data):

    """
    Convierte los datos de precios en una matriz de retornos porcentuales.
    """
    data_returns = data.pct_change()
    return data_returns


def optimize_portfolio(excess_ret_sigma, number_of_assets):
    """
    Optimiza el portafolio minimizando el tracking error.

    Parameters:
    excess_ret_sigma: Matriz de covarianza del exceso de retornos
    number_of_assets: Número de activos en el portafolio

    Returns:
    Las ponderaciones óptimas del portafolio y el tracking error mínimo
    """
    # Crear las variables de decisión
    x = cp.Variable(number_of_assets)

    # Definir las restricciones
    constraints = [
        cp.sum(x) == 1,         # Restricción de presupuesto
        #cp.norm(x, 0) <= 10,     # A lo sumo dos activos seleccionados
        x >= 0,                 # Las ponderaciones deben ser no negativas
        x <= 0.1               # Las ponderaciones no deben superar el 10%
    ]

    # Definir la función objetivo (minimizar el tracking error)
    tracking_error = cp.quad_form(x, excess_ret_sigma)

    # Resolver el problema de optimización
    objective = cp.Minimize(tracking_error)
    problem = cp.Problem(objective, constraints)
    problem.solve()

    if problem.status != cp.OPTIMAL:
      raise RuntimeError(f"Optimization failed with status: {problem.status}")

    # sum(x.value) = 1

    return x.value, problem.value

def matrix_simple_returns(data):

  data = data.to_numpy()
  data = data[1:] / data[:-1] - 1

  return data


tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'PEP', 'AVGO', 'COST',
    'ADBE', 'NFLX', 'TXN', 'INTC', 'QCOM', 'CSCO', 'AMD', 'AMGN', 'HON', 'INTU',
    'AMAT', 'BKNG', 'SBUX', 'VRTX', 'MDLZ', 'GILD', 'ADP', 'ISRG', 'FISV', 'ADI',
    'REGN', 'MRNA', 'LRCX', 'MU', 'ASML', 'SNPS', 'PANW', 'KLAC', 'CTAS', 'NXPI',
    'CHTR', 'IDXX', 'CDNS', 'KDP', 'XEL', 'FTNT', 'CSX', 'MAR', 'MELI', 'WDAY',
    'ORLY', 'AEP', 'MRVL', 'PAYX', 'TEAM', 'MNST', 'ODFL', 'ROST', 'PCAR', 'EXC',
    'VRSK', 'CRWD', 'ABNB', 'LCID', 'AZN', 'WBD', 'SGEN', 'PDD', 'DDOG', 'ZM',
    'DXCM', 'BKR', 'ZS', 'OKTA', 'BIIB', 'SPLK', 'FAST', 'CTSH', 'NTES', 'ANSS',
    'EBAY', 'VRSN', 'LULU', 'ALGN', 'SWKS', 'CDW', 'CHKP', 'SIRI', 'MTCH', 'CPRT',
    'TTWO', 'DLTR', 'FOXA', 'FOX', 'CEG', 'QRVO', 'INVH'
]



get_yahoo_data_table_format(tickers, "^NDX", "2005-04-01", "2024-08-31", "1d", "no","no" )


