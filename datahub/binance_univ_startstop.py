import os

import pandas as pd

from crptmidfreq.config_loc import get_data_folder


class TokenQueryTracker:
    def __init__(self,period='daily'):
        if period=='daily':
            nperiod=''
        else:
            nperiod=period
        self.period=period
        self.log_file = os.path.join(*[get_data_folder(),f"token_query{nperiod}_log.csv"])
        self.query_log = self._load_query_log()

    def _load_query_log(self):
        """Load or create the token query log file."""
        print(f'Reading {self.log_file}')
        if os.path.exists(self.log_file):
            return pd.read_csv(self.log_file, parse_dates=["query_date"])
        return pd.DataFrame(columns=["token", "query_date", "success"])

    def save_query_log(self):
        """Save query log to CSV."""
        self.query_log.to_csv(self.log_file, index=False)

    def log_query(self, token, query_date, success):
        """Log a new query attempt."""
        new_entry = pd.DataFrame([{"token": token, "query_date": query_date, "success": success}])
        self.query_log = pd.concat([self.query_log, new_entry], ignore_index=True)
        self.save_query_log()

    def infer_token_dates(self, token):
        """Infer start_date and end_date for a given token from query logs."""
        token_data = self.query_log[self.query_log["token"] == token]
        if token_data.empty:
            return None, None  # No data available

        successful_queries = token_data[token_data["success"] == 1]["query_date"]

        if successful_queries.empty:
            return None, None  # No successful queries, unknown range

        successful_queries_min=successful_queries.min()
        successful_queries_max=successful_queries.max()
        
        dfloc= token_data[token_data['query_date']<successful_queries_min]
        if dfloc.shape[0]>0:
            actual_min = dfloc['query_date'].max()
        else:
            actual_min=None

        dfloc= token_data[token_data['query_date']>successful_queries_max]
        if dfloc.shape[0]>0:
            actual_max = dfloc['query_date'].min()
        else:
            actual_max=None
        
        return actual_min,actual_max

    def should_query(self, token, query_date):
        """Determine if we should query a token based on inferred start and end dates."""
        query_date=pd.to_datetime(query_date)
        start_date,end_date=self.infer_token_dates(token)
        
        if start_date is None and end_date is None:
            return True  # No prior data, assume we can query

        if start_date and query_date < start_date:
            return False  # Before known start_date

        if end_date and query_date > end_date:
            return False  # After known end_date

        return True  # Query is within the known range


def example_tokenQueryTracker():
    tracker = TokenQueryTracker()

    test_token = "1000BTTCUSD"
    test_dates = pd.date_range("2022-01-01", "2022-06-01")  # Monthly start dates
    known_start = pd.Timestamp("2022-02-01")
    known_end = pd.Timestamp("2022-05-01")

    for date in test_dates:
        if tracker.should_query(test_token, date):
            success = known_start <= date <= known_end  # Simulate Binance query success
            print(f"Querying {test_token} for {date.date()} - {'Success' if success else 'Failure'}")
            tracker.log_query(test_token, date, success)
        else:
            print(f"Skipping {test_token} for {date.date()} - Out of known range")

    # Display inferred dates
    start_date, end_date = tracker.infer_token_dates(test_token)
    print(f"\nToken: {test_token} | Inferred Start: {start_date} | Inferred End: {end_date}")

# python datahub/binance_univ_startstop.py --date_str 2025-03-11
if __name__ == "__main__":
    #example_tokenQueryTracker()
    tracker = TokenQueryTracker(period='monthly')
    r=tracker.should_query('1000BTTCUSDT', '2019-01-01')
    