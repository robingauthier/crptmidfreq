import argparse
import io
import logging
import os
import zipfile

import pandas as pd
import requests

from config_loc import get_data_folder


def download_data(date_str='2024-12-17',
                  kind='bookDepth',
                  ticker='BTCUSDT',
                  instr='futures',
                  ucm='um',
                  klinefreq='1m',
                  period='daily',
                  return_filename=False):
    """
    List of data available on the binance website:

    aggTrades/		
    trades/
    bookTicker/		
    fundingRate/		
    indexPriceKlines/		
    klines/		
    markPriceKlines/		
    premiumIndexKlines/		

    bookTicker is the bid/ask so level 1 only
           update_id  best_bid_price  best_bid_qty  best_ask_price  best_ask_qty        transaction_time              event_time
    0  4307082974693         69903.6         0.462         69903.7         4.419 2024-03-30 00:00:00.002 2024-03-30 00:00:00.008
    1  4307082974976         69903.6         0.362         69903.7         4.419 2024-03-30 00:00:00.005 2024-03-30 00:00:00.011
    2  4307082976403         69900.2         0.047         69900.3         2.582 2024-03-30 00:00:00.014 2024-03-30 00:00:00.021
    3  4307082976763         69900.2         0.047         69900.3         3.293 2024-03-30 00:00:00.016 2024-03-30 00:00:00.023
    4  4307082976780         69900.2         0.047         69900.3         3.693 2024-03-30 00:00:00.016 2024-03-30 00:00:00.023

    I think the 2 main ones are:
    https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-11.zip
    https://data.binance.vision/data/spot/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-11.zip  1 G

    ------------------------------
    https://data.binance.vision/data/futures/um/monthly/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03.zip
    https://data.binance.vision/data/futures/um/monthly/indexPriceKlines/BTCUSDT/1m/BTCUSDT-1m-2024-11.zip 1Mo
    https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1m/BTCUSDT-1m-2024-11.zip 2MO
    https://data.binance.vision/data/futures/um/monthly/premiumIndexKlines/BTCUSDT/1m/BTCUSDT-1m-2024-11.zip 1MO
    https://data.binance.vision/data/futures/um/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-2024-03-31.zip 14k -- stopped
    https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-12-17.zip -- 450k
    https://data.binance.vision/data/futures/um/daily/bookTicker/BTCUSDT/BTCUSDT-bookTicker-2024-03-30.zip -- 83MB not updated

    
    bookDepth data is :
    - updated every 30 seconds
    - bookDepth has 1 update every second with 5 levels only.
    - https://data.binance.vision/data/futures/um/daily/bookDepth/BTCUSDT/BTCUSDT-bookDepth-2024-12-28.zip
                     timestamp  percentage     depth      notional
    28795  2024-12-28 23:59:32           1  1164.189  1.114059e+08
    28796  2024-12-28 23:59:32           2  2580.883  2.482015e+08
    28797  2024-12-28 23:59:32           3  3209.316  3.095250e+08
    28798  2024-12-28 23:59:32           4  3962.634  3.839327e+08
    28799  2024-12-28 23:59:32           5  4595.487  4.469304e+08

    ## aggTrades : trade data marked as Buy/Sell
    aggTrades data is like this:
       agg_trade_id    price  quantity  first_trade_id  last_trade_id           transact_time  is_buyer_maker
    0    2491219245  94258.9     0.005      5781475546     5781475546 2024-12-28 00:00:01.112           False
    1    2491219246  94258.8     0.100      5781475547     5781475548 2024-12-28 00:00:04.862            True
    2    2491219247  94258.9     0.007      5781475549     5781475549 2024-12-28 00:00:04.868           False
    3    2491219248  94258.8     0.025      5781475550     5781475550 2024-12-28 00:00:04.878            True
    4    2491219249  94258.9     0.048      5781475551     5781475553 2024-12-28 00:00:04.908           False

     """
    assert kind in ['trades', 'aggTrades', 'bookTicker', 'bookDepth','klines']
    assert period in ['monthly', 'daily']
    assert instr in ['futures', 'spot']
    root_url = 'https://data.binance.vision/data'
    url_loc = f'/{instr}/{ucm}/{period}/{kind}/{ticker}/{ticker}-{kind}-{date_str}.zip'
    if kind=='klines':
        # there is a klinefreq to add
        url_loc = f'/{instr}/{ucm}/{period}/{kind}/{ticker}/{klinefreq}/{ticker}-{klinefreq}-{date_str}.zip'
    url = root_url + url_loc
    
    # Define the output file path
    output_file = os.path.join(*[get_data_folder(),
                                 'raw_hist',
                                 f'data_{ticker}_{kind}_{date_str}_{instr}_{ucm}_{period}.pq'])
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if return_filename:
        return output_file
    # Check if the data already exists
    if os.path.exists(output_file):
        print(f"Data already exists at {output_file}. Skipping download.")
        return pd.read_parquet(output_file)

    # Download the zip file
    print(f'Calling {url}')
    response = requests.get(url)
    if response.status_code != 200:
        raise ValueError(f"Failed to download: Status code {response.status_code}")

    # Create a zip file object from the downloaded content
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        # Get the CSV filename from the zip
        csv_filename = zip_ref.namelist()[0]
        print(f"Extracting {csv_filename}...")

        # Read the CSV directly from the zip file
        with zip_ref.open(csv_filename) as csv_file:
            # Read the CSV into a pandas DataFrame
            df = pd.read_csv(
                csv_file,
                compression='infer'
            )
    if kind == 'aggTrades':
        df['transact_time'] = pd.to_datetime(df['transact_time'], unit='ms')
    if kind == 'trades':
        df['time'] = pd.to_datetime(df['time'], unit='ms')
    if kind == 'bookTicker':
        df['transaction_time'] = pd.to_datetime(df['transaction_time'], unit='ms')
        df['event_time'] = pd.to_datetime(df['event_time'], unit='ms')

    # Log the file path before saving
    logging.basicConfig(level=logging.INFO)
    logging.info(f'Saving file to {output_file}')

    # Save the DataFrame to a parquet file
    df.to_parquet(output_file, engine='pyarrow')

    metad = {
        'url': url,
        'nbrows': df.shape[0] / 1e6
    }
    print(metad)
    return df


# Syntax to run the download of BTCUSDT futures for 2024-12-21:
# python datahub/binance_hist.py --date_str 2024-12-21 --kind trades --ticker BTCUSDT --instr futures --ucm um --period daily
# python datahub/binance_hist.py --date_str 2024-12-21 --kind klines --ticker all --instr futures --ucm um --period daily
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Binance historical data.')
    parser.add_argument('--date_str', type=str, default='2024-12-17',
                        help='Date string for the data (e.g., 2024-12-21)')
    parser.add_argument('--kind', type=str, default='bookDepth',
                        help='Type of data to download (e.g., trades, aggTrades, bookTicker, bookDepth)')
    parser.add_argument('--ticker', type=str, default='BTCUSDT', help='Ticker symbol (e.g., BTCUSDT)')
    parser.add_argument('--instr', type=str, default='futures', help='Instrument type (e.g., futures, spot)')
    parser.add_argument('--ucm', type=str, default='um', help='UCM type (e.g., um, cm)')
    parser.add_argument('--period', type=str, default='daily', help='Period type (e.g., daily, monthly)')

    args = parser.parse_args()

    df = download_data(
        date_str=args.date_str,
        kind=args.kind,
        ticker=args.ticker,
        instr=args.instr,
        ucm=args.ucm,
        period=args.period
    )
