import yfinance as yf
import pandas as pd
import numpy as np

def load_data(
    ticker: str = "^GSPC",
    start: str = "2010-01-01",
    end: str = "2024-12-31",
    log_returns: bool = True
) -> pd.DataFrame:
    """
    Parameters
    ----------
    ticker : str
        Ticker symbol (default: '^GSPC' for S&P 500).
    start : str
        Start date (YYYY-MM-DD).
    end : str
        End date (YYYY-MM-DD).
    log_returns : bool
        If True, compute log returns; else percentage returns.

    Returns
    -------
    pd.DataFrame
        Columns: ['Price', 'Returns'] indexed by Date.
    """

    # Download historical data
    data = yf.download(ticker, start=start, end=end)

    # Flatten multi-index (some tickers return MultiIndex columns)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [c[0] for c in data.columns]

    # Use adjusted close if available
    price_col = "Adj Close" if "Adj Close" in data.columns else "Close"
    data = data.rename(columns={price_col: "Price"})

    # Compute returns
    if log_returns:
        data["Returns"] = np.log(data["Price"]).diff()
    else:
        data["Returns"] = data["Price"].pct_change()

    # Drop missing rows
    data = data.dropna().reset_index()

    # Keep only relevant columns
    data = data[["Date", "Price", "Returns"]]

    return data


if __name__ == "__main__":
    df = load_data("^GSPC")
    print(df.head())
