import os
import pickle
import pandas as pd
import numpy as np
from numba import njit
from numba import types
from numba.typed import Dict
from crptmidfreq.stepper.base_stepper import BaseStepper
from crptmidfreq.utils.common import get_logger
from crptmidfreq.utils.bktester import bktest_stats
from crptmidfreq.utils.bktester import get_daily_stats
from crptmidfreq.config_loc import get_analysis_folder
import matplotlib
import matplotlib.pyplot as plt

# Quite similar to the featurelib/timeclf.py

logger = get_logger()


class BktestStepper(BaseStepper):
    """Only way to backtest and keep information"""

    def __init__(self, folder='', name=''):
        """
        """
        super().__init__(folder, name)
        self.dailypnl = pd.DataFrame()
        self.statsdf = pd.DataFrame()
        self.with_txt = True
        with_plot = True
        if with_plot:
            self.save_graph_path = os.path.join(get_analysis_folder(), f'bktest_{self.name}_COLNAME.png')
        else:
            self.save_graph_path = None

    def save(self):
        self.save_utility()

    @classmethod
    def load(cls, folder, name, lookback=300, minlookback=100,
             fitfreq=10, gap=1, model_gen=None, with_fit=True,
             featnames=[]):
        """Load instance from saved state or create new if not exists"""
        return BktestStepper.load_utility(cls, folder=folder, name=name)

    def display_stats(self):
        lr = []
        df = self.dailypnl
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
                'ypred_std': np.nan
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
                fig, ax = plt.subplots(figsize=(10, 6))
                daily_dt_f = pd.to_datetime(dt*1e3)
                ax.plot(daily_dt_f, cum_net)
                ax.set_title(f'Cumulative Net PnL {col}')
                tsave_graph_path = self.save_graph_path.replace('COLNAME', col)
                plt.savefig(tsave_graph_path)
                plt.close(fig)
                logger.info(f'Bktest Graph saved to {tsave_graph_path}')

        rptdf = pd.DataFrame(lr)
        rptdf1 = rptdf[['name', 'col', 'sr', 'rpt', 'mdd', 'rog', 'avg_gmv', 'ann_pnl', 'cnt']].round(2)
        if self.with_txt:
            print('Gross P&L Stats:')
            print(rptdf1)
        return rptdf1

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
            dfpnl = pd.DataFrame(dpnl)
            dfpnl['colname'] = sig_col
            ldf += [dfpnl]
        df = pd.concat(ldf, axis=0, sort=False)
        if self.dailypnl.shape[0] > 0:
            self.dailypnl = pd.concat([self.dailypnl, df], axis=0)
        else:
            self.dailypnl = df
        self.statsdf = self.display_stats()
