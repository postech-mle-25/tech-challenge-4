import argparse
from datetime import date, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.data.data_loader import StockDataLoader

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOG", "META", "AMZN",
    "NVDA", "TSLA", "ITUB4.SA", "PETR4.SA", "VALE3.SA"
]

def fetch_ticker(ticker: str, start: str, end: str) -> dict:
    """Fetches a single ticker and returns result."""
    try:
        loader = StockDataLoader(ticker, start, end)
        df = loader.fetch_data()
        return {
            "ticker": ticker,
            "success": True,
            "rows": len(df),
            "source": loader.get_source()
        }
    except Exception as e:
        return {
            "ticker": ticker,
            "success": False,
            "error": str(e)
        }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None,
                        help="Substitui completamente a lista padrão")
    parser.add_argument("--add-tickers", nargs="*", default=[],
                        help="Adiciona tickers à lista padrão")
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--workers", type=int, default=5,
                        help="Número de threads paralelas (padrão: 5)")
    args = parser.parse_args()

    # Se --tickers foi passado, usa ele. Senão, usa DEFAULT + add-tickers
    if args.tickers is not None:
        tickers = args.tickers
    else:
        tickers = DEFAULT_TICKERS + args.add_tickers

    end = date.today()
    start = end - timedelta(days=365 * args.years)

    print(f"Warmup cache: {len(tickers)} tickers | período {start} → {end}")
    print(f"Usando {args.workers} threads paralelas\n")

    # Paralelização com ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(fetch_ticker, tk, str(start), str(end)): tk
            for tk in tickers
        }

        for future in as_completed(futures):
            result = future.result()
            if result["success"]:
                print(f"  ✅ {result['ticker']}: {result['rows']} linhas | fonte={result['source']}")
            else:
                print(f"  ❌ {result['ticker']}: {result['error']}")

if __name__ == "__main__":
    main()