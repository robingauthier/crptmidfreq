import argparse
import io
import logging
import os
import zipfile
import xmltodict
import pandas as pd
import requests

from config_loc import get_data_folder

def extract_base_loc(base,sym,restype,adj=''):
    """restype = 1 : base   (ex :USDT,BTC)
       restype = 2 : ticker (ex :EOS,FFT)
       restype = 3 : is leveraged (UP/DOWN...)
       restype=4 : sorting id
    """
    orderd={
        'USDT': 1,
        'BTC': 2,
        'ETH':3,
        'BUSD': 4,
        'BNB':5,
        'USDC':6,
        'TUSD':7,
        'FDUSD':8,
        'USD':9,
    }
    if restype == 1:
        return base
    elif restype == 2:
        return sym.replace(adj+base, '')
    elif restype == 3:
        return adj
    else:
        if adj!='':
            return 12
        elif base in orderd.keys():
            return orderd[base]
        else:
            return 10
def extract_base(sym,restype=1):
    bases=['FDUSD','BUSD','USDT','USDC','TUSD','USD',
           'BNB','ETH','BTC','BIDR','PAX','BRL',
           'EUR','RUB','TRY','AUD','GBP','XRP',
           'NGN','TRX']
    for base in bases:
        if  sym.endswith('UP'+base):
            return extract_base_loc(base,sym,restype=restype,adj='UP')
        if sym.endswith('DOWN'+base):
            return extract_base_loc(base,sym,restype=restype,adj='DOWN')
        if sym.endswith('BULL'+base):
            return extract_base_loc(base,sym,restype=restype,adj='BULL')
        if sym.endswith('BEAR' + base):
            return extract_base_loc(base,sym,restype=restype,adj='BEAR')
        if sym.endswith(base):
            return extract_base_loc(base,sym,restype=restype)
        
def download_univ(
    date_str='2024-12-17'
    ):
    """
    binance does not have a PIT universe
    
                                                Prefix       kind     PrefixTemp            sym  base     ticker isupdown  orderid
    0             data/spot/monthly/klines/1000CATBNB/       spot     1000CATBNB     1000CATBNB   BNB    1000CAT               5.0
    1           data/spot/monthly/klines/1000CATFDUSD/       spot   1000CATFDUSD   1000CATFDUSD   USD  1000CATFD              10.0
    2             data/spot/monthly/klines/1000CATTRY/       spot     1000CATTRY     1000CATTRY   TRY    1000CAT              10.0
    """        

    output_file = os.path.join(*[get_data_folder(),
                                'universe',
                                f'universe_{date_str}.pq'])
    
    # make sure folder structure is there
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_file):
        return pd.read_parquet(output_file)
    
    date_str = pd.to_datetime('now').strftime('%Y-%m-%d')
    output_file = os.path.join(*[get_data_folder(),
                                'universe',
                                f'universe_{date_str}.pq'])
    if os.path.exists(output_file):
        return pd.read_parquet(output_file)
    
    root_url = 'https://s3-ap-northeast-1.amazonaws.com/data.binance.vision?delimiter=/&prefix=data/'
    kindd={
        'spot': 'spot/monthly/klines/', # on the SPOT, it goes from A to M . Hence it cuts the results to 1000 rows.
        'spot2': 'spot/monthly/klines/&marker=data%2Fspot%2Fmonthly%2Fklines%2FMIRBUSD%2F',#page 2 on the spot history list
        'future_um': 'futures/um/monthly/klines/',
        'future_cm': 'futures/cm/monthly/klines/',
        }
    lres=[]
    for kind,val in kindd.items():
        url=root_url+val
        raw=requests.get(url)
        dd=xmltodict.parse(raw.content)
        res=pd.DataFrame(dd['ListBucketResult']['CommonPrefixes'])
        res['kind']=kind.replace('2','')
        lres+=[res]
    df = pd.concat(lres,axis=0,sort=False)
    df['PrefixTemp']=df['Prefix'].str[:-1]
    df['PrefixTemp']=df['PrefixTemp'].str.replace('data/spot/monthly/klines/','')
    df['PrefixTemp'] = df['PrefixTemp'].str.replace('data/futures/cm/monthly/klines/', '')
    df['PrefixTemp'] = df['PrefixTemp'].str.replace('data/futures/um/monthly/klines/', '')
    df['sym_original']=df['PrefixTemp']
    df['sym'] = df['PrefixTemp'].str.replace('_PERP', '')
    # Handling the case : XRPUSD_250328
    df['sym'] = df['sym'].str.extract(r'(.*)_\d{6}')[0].fillna(df['PrefixTemp'])
    df['base']=df['sym'].apply(lambda x:extract_base(x,1))
    df['ticker'] = df['sym'].apply(lambda x: extract_base(x, 2))
    df['isupdown'] = df['sym'].apply(lambda x: extract_base(x, 3))
    df['orderid'] = df['sym'].apply(lambda x: extract_base(x, 4)).fillna(100)
    df=df.drop(['PrefixTemp'],axis=1)
    assert df.shape[0]>0
    df.to_parquet(output_file)
    return df

# python datahub/binance_univ.py --date_str 2025-03-11
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Binance Universe data.')
    parser.add_argument('--date_str', type=str, default='2025-03-11',
                        help='Date string for the data (e.g., 2024-12-21)')
    args = parser.parse_args()

    df = download_univ(date_str=args.date_str)

