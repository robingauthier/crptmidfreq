

import requests
import csv
import argparse
import os
import pandas as pd
import json
from crptmidfreq.config_loc import get_data_folder
# there is an unofficial API 

def download_univ2(date_str='2025-03-11'):
    """
    https://stackoverflow.com/questions/68836406/binance-api-list-all-symbols-with-their-names-from-a-public-endpoint
    
    Fetch all asset data from Binance's public API endpoint and save it to CSV.
        id assetCode        assetName  trading  delisted    etf                            tags    createTime  isLegalMoney  commissionRate  assetDigit
    0   10       BNB              BNB     True     False  False  Layer1_Layer2,BSC,pos,bnbchain  1.645070e+12         False               0           8
    1  208     BCHSV  Bitcoin Cash SV    False      True  False                             NaN  1.651820e+12         False               0           8
    2  374       CHR          Chromia     True     False  False           Layer1_Layer2,NFT,RWA  1.588830e+12         False               0           8
    
    
    
    Complete list of available information is:
    {'id': '374',
    'assetCode': 'CHR',
    'assetName': 'Chromia',
    'unit': '',
    'commissionRate': 0.0,
    'freeAuditWithdrawAmt': 0.0,
    'freeUserChargeAmount': 12797540.0,
    'createTime': 1588833159000,
    'test': 0,
    'gas': 0.0,
    'isLegalMoney': False,
    'reconciliationAmount': 0.0,
    'seqNum': '0',
    'chineseName': 'Chromia',
    'cnLink': '',
    'enLink': '',
    'logoUrl': 'https://bin.bnbstatic.com/images/20200507/1ca0e8c6-e5bc-4cf3-94ee-2a5621afff67.png',
    'fullLogoUrl': 'https://bin.bnbstatic.com/images/20200507/1ca0e8c6-e5bc-4cf3-94ee-2a5621afff67.png',
    'supportMarket': None,
    'feeReferenceAsset': '',
    'feeRate': None,
    'feeDigit': 8,
    'assetDigit': 8,
    'trading': True,
    'tags': ['Layer1_Layer2', 'NFT', 'RWA'],
    'plateType': 'false',
    'etf': False,
    'isLedgerOnly': False,
    'delisted': False,
    'preDelist': False,
    'pdTradeDeadline': None,
    'pdDepositDeadline': None,
    'pdAnnounceUrl': None,
    'tagBits': '0',
    'oldAssetCode': None,
    'newAssetCode': None,
    'swapTag': 'no',
    'swapAnnounceUrl': None}
    """
    output_file = os.path.join(*[get_data_folder(),
                                'universe',
                                f'universe2_{date_str}.csv'])
    
    # make sure folder structure is there
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)
    
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    
    date_str = pd.to_datetime('now').strftime('%Y-%m-%d')
    output_file = os.path.join(*[get_data_folder(),
                                'universe',
                                f'universe2_{date_str}.csv'])
    if os.path.exists(output_file):
        return pd.read_csv(output_file)
    
    url = "https://www.binance.com/bapi/asset/v2/public/asset/asset/get-all-asset"
    response = requests.get(url)
    response.raise_for_status()  # Raises an HTTPError if the request returned an unsuccessful status code
    data = response.json()
    #import pdb;pdb.set_trace()

    # saving down the full data in case    
    with open(output_file.replace('.pq','.json'),'wt') as f:
        f.write(json.dumps(data))
    
    # The actual asset list lives in data["data"]
    assets = data.get("data", [])
    if not assets:
        print("No assets found in the response.")
        return

    # Choose which fields you want in your CSV
    fieldnames = [
        "id",
        "assetCode",
        "assetName",
        "trading",
        "delisted",
        "etf",
        "tags",
        "createTime",
        "isLegalMoney",
        "commissionRate",
        "assetDigit",
    ]

    # Write to CSV
    with open(output_file, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for asset in assets:
            # For safety, use .get() to avoid KeyErrors if a field is missing
            writer.writerow({
                "id": asset.get("id"),
                "assetCode": asset.get("assetCode"),
                "assetName": asset.get("assetName"),
                "trading": asset.get("trading"),
                "delisted": asset.get("delisted"),
                "tags": ",".join(asset.get("tags", [])) if asset.get("tags") else "",
                "createTime": pd.to_datetime(asset.get("createTime",np.nan)*1e6),
                "isLegalMoney": asset.get("isLegalMoney"),
                "commissionRate": asset.get("commissionRate"),
                "assetDigit": asset.get("assetDigit"),
            })

    return pd.read_csv(output_file)

# ipython -i datahub/binance_univ2.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Download Binance Universe data.')
    parser.add_argument('--date_str', type=str, default='2025-03-11',
                        help='Date string for the data (e.g., 2024-12-21)')
    args = parser.parse_args()

    df = download_univ2(date_str=args.date_str)

