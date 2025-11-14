import argparse
from datetime import date, timedelta

from src.data.data_loader import StockDataLoader

DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOG", "META", "AMZN",
    "NVDA", "TSLA", "ITUB4.SA", "PETR4.SA", "VALE3.SA"
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=DEFAULT_TICKERS)
    parser.add_argument("--years", type=int, default=8)
    parser.add_argument("--allow_synthetic", action="store_true",
                        help="Permite sintético como último recurso (não será cacheado).")
    args = parser.parse_args()

    end = date.today()
    start = end - timedelta(days=365 * args.years)

    print(f"Warmup cache: {len(args.tickers)} tickers | período {start} → {end}")
    for tk in args.tickers:
        try:
            loader = StockDataLoader(tk, str(start), str(end), allow_synthetic=args.allow_synthetic)
            df = loader.fetch_data()
            print(f"  ✅ {tk}: {len(df)} linhas | fonte={loader.get_source()}")
        except Exception as e:
            print(f"  ❌ {tk}: {e}")

if __name__ == "__main__":
    main()
