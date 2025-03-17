import pandas as pd
import argparse
import random
import time
import os
import duckdb
from datahub.binance_univ import download_univ
from datahub.binance_hist import download_data
from datahub.binance_univ_startstop import  TokenQueryTracker
from config_loc import get_data_folder
from config_loc import get_data_db_folder
from dateutil.relativedelta import relativedelta

def stats_on_univ(dfuniv):
    # Stats on dfuniv
    print('stats on dfuniv')
    print(dfuniv['kind'].value_counts()) # mainly SPOT
    print(dfuniv['base'].value_counts()) # USDT and then BTC
    
def main(args):
    assume_duckdb_updated=True
    
    con = duckdb.connect(os.path.join(get_data_db_folder(),"my_database.db"))
    con.execute("PRAGMA enable_checkpoint_on_shutdown;")
    
    # Ensure table exists
    assert args.kind=='klines'
    if args.kind=='klines':
        con.execute("""
    CREATE TABLE IF NOT EXISTS klines (
        open_time TIMESTAMP, 
        close_time TIMESTAMP, 
        dscode TEXT,
        open FLOAT, 
        high FLOAT, 
        low FLOAT, 
        close FLOAT, 
        volume FLOAT, 
        quote_volume FLOAT, 
        count INTEGER, 
        taker_buy_volume FLOAT, 
        taker_buy_quote_volume FLOAT, 
        ignore INTEGER
    )
""")

    tracker = TokenQueryTracker(period=args.period)
 
    dfuniv = download_univ()
    dfuniv = dfuniv.loc[lambda x:x['kind']=='future_um']
    dfuniv = dfuniv.loc[lambda x:x['base']=='USDT']
    #dfuniv = dfuniv[dfuniv['sym_original'].isin(['BTCUSDT','ETHUSDT'])]
    
    if args.period=='daily':
        dts = pd.date_range(args.sdate_str,args.edate_str).tolist()
    else:
        dts = pd.date_range(args.sdate_str,args.edate_str,freq='ME').tolist()
    random.shuffle(dts)
    
    for dt in dts:
        for id,row in dfuniv.iterrows():
            ticker=row['sym_original']
            dt_str=dt.strftime('%Y-%m-%d')
            
            # if we know this ticker no longer exists at this point
            if not tracker.should_query(ticker, dt):
                continue
            
            # Adding the ticker            
            if row['kind']=='future_um':
                ticker_db = ticker
            else:
                ticker_db=ticker+'_'+row['kind']

            if args.period=='daily':
                # check if the data is there already or not
                query=f'''
                SELECT COUNT(open_time) AS nb_rows FROM klines 
                WHERE dscode='{ticker_db}' AND 
                CAST(open_time AS DATE) = '{dt_str}';
                '''
                nb_existing_rows=con.execute(query=query).fetchone()[0]
            else:
                start_month = pd.to_datetime(dt.strftime('%Y-%m')+'-01')
                end_month = start_month+relativedelta(months=1)
                end_month = end_month-relativedelta(days=1)
                start_month_str=start_month.strftime('%Y-%m-%d')
                end_month_str=end_month.strftime('%Y-%m-%d')
                query=f'''
                SELECT COUNT(open_time) AS nb_rows FROM klines 
                WHERE dscode='{ticker_db}' AND 
                CAST(open_time AS DATE) >= '{start_month_str}'
                AND 
                CAST(open_time AS DATE) <= '{end_month_str}';
                '''
                nb_existing_rows=con.execute(query=query).fetchone()[0]                
                
            if nb_existing_rows>0:
                continue
            
            data_args=dict(                    
                    date_str=dt.strftime('%Y-%m-%d'),
                    kind=args.kind,
                    ticker=ticker,
                    instr='futures',
                    ucm='um',
                    period=args.period, # on ETHUSDT it starts on 2019-12-31
            )   
            raw_data_fname=download_data(**data_args,return_filename=True)
            if os.path.exists(raw_data_fname) and assume_duckdb_updated:
                continue
            
            # Try to download the data
            try:
                df = download_data(**data_args)
            except ValueError as e:
                # Handling the case the url does not exists / no such file
                print(e)
                df=None
            
            # making sure we track what we tried
            success = df is not None
            tracker.log_query(ticker,dt,success=success)
            if not success:
                continue
            if df.shape[0]==0:
                continue
            
            # Insert into DuckDB
            if "open_time" not in df.columns:
                print(df.head())
                import pdb;pdb.set_trace()
            df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", errors="coerce")
            df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", errors="coerce")
            df['dscode'] = ticker_db

            start_insert_time = pd.to_datetime('now')
            
            # column order matters
            wcols=['open_time','close_time', 'dscode','open', 
                'high', 'low', 'close', 'volume', 
                'quote_volume', 'count', 
                'taker_buy_volume', 'taker_buy_quote_volume', 
                'ignore']
            df[wcols].to_csv("temp_data.csv", index=False)  # Save to a temporary file
            con.execute("COPY klines FROM 'temp_data.csv' (AUTO_DETECT TRUE);")
            end_insert_time = pd.to_datetime('now')
            insert_time = end_insert_time-start_insert_time
            print(insert_time)
            
            time.sleep(0.5)
    con.close()
    
# python datahub/binance_hist_wrap.py --sdate_str 2023-01-21 
# python  datahub/binance_hist_wrap.py --sdate_str 2020-01-01 --edate_str 2025-01-01 --period monthly
if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Download Binance historical data.')
    parser.add_argument('--sdate_str', type=str, default='2025-01-01',
                        help='Date string for the data (e.g., 2024-12-21)')
    parser.add_argument('--edate_str', type=str, default='2025-03-11',
                        help='Date string for the data (e.g., 2024-12-21)')
    parser.add_argument('--kind', type=str, default='klines',
                        help='Type of data to download (e.g., trades, aggTrades, bookTicker, bookDepth)')
    parser.add_argument('--period', type=str, default='daily',
                        help='Type of data to download (e.g., trades, aggTrades, bookTicker, bookDepth)')

    args = parser.parse_args()
    main(args=args)