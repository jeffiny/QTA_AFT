class Operators:
    # cross section operators
    @staticmethod
    def _cs_add(df, cols):
        for c in itertools.combinations(cols, 2):
            df = (df
                .assign(**{f'cs_add_{c[0]}_{c[1]}': lambda x: x[c[0]] + x[c[1]]})
            )
        return df

    @staticmethod
    def _cs_zscore_add(df, cols):
        for col in cols:
            df[f"cs_zscore_{col}"] = (df[col] - df.groupby(['time_id'])[col].transform('mean')) / df.groupby(['time_id'])[col].transform('std')
        for c in itertools.combinations(cols, 2):
            df = (df
                .assign(**{f'cs_add_{c[0]}_{c[1]}': lambda x: x['cs_zscore_'+c[0]] + x['cs_zscore_'+c[1]]})
            )
        return df
    
    @staticmethod
    def _cs_rank_add(df, cols):
        for col in cols:
            df[f"cs_rank_{col}"] = df.groupby(['time_id'])[col].rank(pct=True)
        for c in itertools.combinations(cols, 2):
            df = (df
                .assign(**{f'cs_add_{c[0]}_{c[1]}': lambda x: x['cs_rank_'+c[0]] + x['cs_rank_'+c[1]]})
            )
        return df
    
    @staticmethod
    def _cs_multiply(df, cols):
        for c in itertools.combinations(cols, 2):
            df = (df
                .assign(**{f'cs_multiply_{c[0]}_{c[1]}': lambda x: x[c[0]] * x[c[1]]})
            )
        return df
    
    @staticmethod
    def _cs_divide(df, cols):
        for c in itertools.combinations(cols, 2):
            assert (df[c[1]] != 0).all()
            df = (df
                .assign(**{f'cs_divide_{c[0]}_{c[1]}': lambda x: x[c[0]] / x[c[1]]})
            )
        return df


    # time series operators
    @staticmethod
    def _ts_mean(df, cols, windows=[]):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_mean_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        return df

    @staticmethod
    def _ts_std(df, cols, windows=[]):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_std_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
        return df
    
    @staticmethod
    def _ts_corr(df, cols, windows=[]): #so slow
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for c in itertools.combinations(cols, 2):
            for window in windows:
                df[f"ts_corr_{c[0]}_{c[1]}_{window}"] = df.groupby(['date_stock']).apply(
                    lambda x: x[[c[0], c[1]]].rolling(window=window, min_periods=1).corr().iloc[0::2,1].reset_index(level=1, drop=True)
                    ).reset_index(level=0, drop=True)
        return df
    
    @staticmethod
    def _ts_zscore(df, cols, windows=[]):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_zscore_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: (x-x.rolling(window=window, min_periods=1).mean())/x.rolling(window=window, min_periods=1).std())
        return df
    
    @staticmethod
    def _ts_rank(df, cols, windows=[]):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_rank_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).rank(pct=True))
        return df

    @staticmethod
    def _ts_slope(df, cols, windows=[]): # so slow
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_slope_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        lambda y: np.polyfit(range(len(y)), y, 1)[0]
                    )
                )
        return df
    
    @staticmethod
    def _ts_diff(df, cols, windows=[]):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_diff_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.diff(periods=window))
        return df