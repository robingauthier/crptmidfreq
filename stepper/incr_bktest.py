import os
import pandas as pd
import numpy as np
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.bktester import bktest_stats
from crptmidfreq.utils.bktester import get_daily_stats
from crptmidfreq.config_loc import get_analysis_folder
import matplotlib.pyplot as plt

# Quite similar to the featurelib/timeclf.py

logger = get_logger()


def try_to_save_png(tsave_graph_path):
    try:
        plt.savefig(tsave_graph_path)
        logger.info(f'Bktest Graph saved to {tsave_graph_path}')
    except Exception as e:
        try_to_save_png(tsave_graph_path.replace('.png', '_bis.png'))
    plt.close()


class BktestStepper(BaseStepper):
    """Only way to backtest and keep information"""

    def __init__(self, folder='', name=''):
        """
        """
        super().__init__(folder, name)
        self.dailypnl = pd.DataFrame()  # minute/or any dataset unit
        self.statsdf = pd.DataFrame()
        self.dailypnl2 = pd.DataFrame()  # actual date 2024-01-01, 2024-01-02
        self.with_txt = True
        with_plot = True
        if with_plot:
            self.save_graph_path = os.path.join(get_analysis_folder(), f'{self.name}_COLNAME.png')
        else:
            self.save_graph_path = None

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name):
        """Load instance from saved state or create new if not exists"""
        return BktestStepper.load_utility(cls, folder=folder, name=name)

    def display_stats(self):
        lr = []
        df = self.dailypnl
        if df.shape[0] == 0:
            return pd.DataFrame()
        for col, dfloc in df.groupby('colname'):
            if dfloc.shape[0] == 0:
                continue
            rd = {
                'name': self.name,
                'col': col,
                'cnt': np.nan,
                'sr': np.nan,
                'mdd': np.nan,
                'rpt': np.nan,
                'rog': np.nan,
                'ann_pnl': np.nan,
                'factor': np.nan,
                'sigma': np.nan,
                'srs': None,
                'ypred_qtiles': None,
                'trd_cost': np.nan,
                'comm_cost': np.nan,
                'fin_cost': np.nan,
                'net_sr2': np.nan,
                'avg_pos': np.nan,
                'avg_gmv': np.nan,
                'med_gmv': np.nan,
                'ypred_std': np.nan,
                'sdate': np.nan,  # start date
                'edate': np.nan,  # end date
            }
            dt = dfloc['daily_dt']
            tot_pnl = dfloc['daily_gross_pnl']
            tot_trd = dfloc['daily_trd']
            tot_gmv = dfloc['daily_gmv']
            rd = get_daily_stats(dt, tot_pnl, tot_trd, tot_gmv, rd=rd, suf='')
            lr += [rd]

            # --------------------------
            # Optional: Save Graphs and Daily PnL
            # --------------------------
            if self.save_graph_path is not None:
                # Plot cumulative net pnl (using daily_net from pandas aggregation)
                cum_net = np.cumsum(tot_pnl)
                tot_gmv_ewm = tot_gmv.ewm(halflife=60).mean()
                fig, ax = plt.subplots(figsize=(10, 6))
                daily_dt_f = pd.to_datetime(dt*1e3)
                ax.plot(daily_dt_f, cum_net, label='Cum PnL')
                ax2 = ax.twinx()
                ax2.plot(daily_dt_f, tot_gmv_ewm, alpha=0.3, label='Gross Delta')
                ax.set_title(f'Cumulative Net PnL {col}')
                ax.legend()
                ax2.legend()
                tsave_graph_path = self.save_graph_path.replace('COLNAME', col)
                try_to_save_png(tsave_graph_path)

        rptdf = pd.DataFrame(lr)
        rptdf1 = rptdf[['name', 'col', 'sr', 'rpt', 'mdd', 'rog', 'avg_gmv', 'ann_pnl', 'cnt']].round(2)
        if self.with_txt:
            print('Gross P&L Stats:')
            print(rptdf1.sort_values('sr'))
        return rptdf1

    def compute_daily_stats(self):
        dailypnl = self.dailypnl.copy()
        dailypnl['daily_dt'] = pd.to_datetime(dailypnl['daily_dt']*1e3)
        dailypnl['date'] = pd.to_datetime(dailypnl['daily_dt'].dt.strftime('%Y-%m-%d'))
        dailypnl2 = dailypnl\
            .groupby(['date', 'colname'])\
            .agg({'daily_net_pnl': 'sum',
                  'daily_trd': 'sum',
                  'daily_gmv': 'sum'})\
            .reset_index()
        return dailypnl2

    def update(self, featd):
        """

        """
        assert 'forward_fh1' in featd.keys()
        assert 'wgt' in featd.keys()
        assert 'dtsi' in featd.keys()
        assert 'dscode' in featd.keys()
        sig_cols = [x for x in featd.keys() if x.startswith('sig_')]
        ldf = []
        for sig_col in sig_cols:
            dpnl = bktest_stats(
                featd['dtsi'],
                featd['dscode'],
                featd[sig_col],
                featd['forward_fh1'],
                featd['wgt'],
                str(sig_col),  # name
                out_dailypnl=True,
            )
            if dpnl is None:
                continue
            if dpnl['daily_dt'].shape[0] == 0:
                continue
            dfpnl = pd.DataFrame(dpnl)
            dfpnl['colname'] = sig_col
            ldf += [dfpnl]
        if len(ldf) > 0:
            df = pd.concat(ldf, axis=0, sort=False)
            if self.dailypnl.shape[0] > 0:
                self.dailypnl = pd.concat([self.dailypnl, df], axis=0)
            else:
                self.dailypnl = df
            self.dailypnl = self.dailypnl.sort_values('daily_dt')
        self.dailypnl2 = self.compute_daily_stats()
        self.statsdf = self.display_stats()
