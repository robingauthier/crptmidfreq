import pandas as pd

from crptmidfreq.season.hols import generate_event_calendar

pd.set_option('display.max_rows', 500)


def event_distances(start_date, end_date, win=60):
    """
    event_df : DataFrame with columns ['date','evtname','evtimp']
    start_date, end_date : strings or datetime-like

    Returns a DataFrame with:
       date,
       dist_to_next_ALL, dist_since_last_ALL,
       dist_to_next_High, dist_since_last_High,
       dist_to_next_Mid,  dist_since_last_Mid,
       ... for any other evtimp in event_df.
    """
    event_df = generate_event_calendar(start_date.year, end_date.year)
    event_df = event_df.drop_duplicates(subset=['date'])

    # 1) daily index
    idx = pd.date_range(start_date, end_date, freq='D')
    rdf = pd.DataFrame({'date': idx})

    # 2) build for “All” events
    todel = []
    kind = 'all'
    for kind in ['all']+event_df['evtimp'].unique().tolist():
        if kind == 'all':
            event_df_loc = event_df.copy()
        else:
            event_df_loc = event_df.loc[lambda x:x['evtimp'] == kind]
        all_evts = event_df_loc[['date', 'evtname']].rename(columns={'evtname': f'evt_{kind}_prev'})
        all_evts[f'date_evt_{kind}_prev'] = all_evts['date']
        rdf = pd.merge_asof(rdf, all_evts, on='date', direction='backward')
        rdf[f'dist_since_last_{kind}'] = (rdf['date']-rdf[f'date_evt_{kind}_prev']).dt.days
        rdf[f'dist_since_last_{kind}'] = rdf[f'dist_since_last_{kind}'].clip(lower=-1*win, upper=win)

        all_evts = event_df_loc[['date', 'evtname']].rename(columns={'evtname': f'evt_{kind}_next'})
        all_evts[f'date_evt_{kind}_next'] = all_evts['date']
        rdf = pd.merge_asof(rdf, all_evts, on='date', direction='forward')

        rdf[f'dist_to_next_{kind}'] = (rdf['date']-rdf[f'date_evt_{kind}_next']).dt.days
        rdf[f'dist_to_next_{kind}'] = rdf[f'dist_to_next_{kind}'].clip(lower=-1*win, upper=win)

        todel += [f'date_evt_{kind}_next', f'date_evt_{kind}_prev']
    rdf.drop(todel, axis=1, inplace=True)
    return rdf
