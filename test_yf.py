import yfinance as yf
import multiprocessing

def fetch(sym):
    yf.config.use_cache = False
    data = yf.Ticker(sym).history(period='1d')
    return sym, data.empty

if __name__ == "__main__":
    yf.config.use_cache = False
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'PYPL', 'ADBE', 'NFLX']
    with multiprocessing.Pool(5) as p:
        results = p.map(fetch, symbols)
    print(results)
