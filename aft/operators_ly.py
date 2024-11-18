import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
'''
Editor: Yang Lan
Time: 2024/11/2
Content: Cross Section Operators & Time series Operators 
         注释了和Zhifan重合的部分
'''
class Operators:
    # cross section operators
    # @staticmethod
    # def _cs_add(df, cols):
    #     for c in itertools.combinations(cols, 2):
    #         df = df.assign(**{f'cs_add_{c[0]}_{c[1]}': lambda x: x[c[0]] + x[c[1]]})
    #     return df
    
    # @staticmethod
    # def _cs_subtract(df, cols):
    #     for c in itertools.combinations(cols, 2):
    #         df = df.assign(**{f'cs_subtract_{c[0]}_{c[1]}': lambda x: x[c[0]] - x[c[1]]})
    #     return df
    
    # @staticmethod
    # def _cs_multiply(df, cols):
    #     for c in itertools.combinations(cols, 2):
    #         df = df.assign(**{f'cs_multiply_{c[0]}_{c[1]}': lambda x: x[c[0]] * x[c[1]]})
    #     return df
    
    # @staticmethod
    # def _cs_divide(df, cols):
    #     for c in itertools.combinations(cols, 2):
    #         df = df.assign(**{f'cs_divide_{c[0]}_{c[1]}': lambda x: x[c[0]] / x[c[1]] if x[c[1]] != 0 else None})
    #     return df

    @staticmethod
    def _cs_max(df, cols):
        df['cs_max'] = df[cols].max(axis=1)
        return df

    @staticmethod
    def _cs_min(df, cols):
        df['cs_min'] = df[cols].min(axis=1)
        return df

    @staticmethod
    def _cs_range(df, cols):
        df['cs_range'] = df[cols].max(axis=1) - df[cols].min(axis=1)
        return df

    @staticmethod
    def _cs_mean(df, cols):
        df['cs_mean'] = df[cols].mean(axis=1)
        return df

    @staticmethod
    def _cs_std(df, cols):
        df['cs_std'] = df[cols].std(axis=1)
        return df

    @staticmethod
    def _cs_weighted_sum(df, cols, weights):
        df['cs_weighted_sum'] = sum(df[col] * weight for col, weight in zip(cols, weights))
        return df

    @staticmethod
    def _cs_exponential_moving_average(df, cols, span):
        for col in cols:
            df[f'cs_ema_{col}'] = df[col].ewm(span=span, adjust=False).mean()
        return df

    @staticmethod
    def _cs_return(df, price_col, period):
        df[f'cs_return_{period}'] = df[price_col].pct_change(periods=period)
        return df
    @staticmethod
    def _group_mean(df, group_cols, target_col):
        return df.groupby(group_cols)[target_col].mean().reset_index(name=f'group_mean_{target_col}')

    @staticmethod
    def _group_std(df, group_cols, target_col):
        return df.groupby(group_cols)[target_col].std().reset_index(name=f'group_std_{target_col}')

    @staticmethod
    def _group_max(df, group_cols, target_col):
        return df.groupby(group_cols)[target_col].max().reset_index(name=f'group_max_{target_col}')

    @staticmethod
    def _group_min(df, group_cols, target_col):
        return df.groupby(group_cols)[target_col].min().reset_index(name=f'group_min_{target_col}')

    @staticmethod
    def _calculate_residuals(df, group_cols, target_col, predictor_col):
        # 计算回归残差
        residuals = []
        for _, group in df.groupby(group_cols):
            X = sm.add_constant(group[predictor_col])  # 加上常数项
            y = group[target_col]
            model = sm.OLS(y, X).fit()
            residuals.append(model.resid)
        df['residuals'] = np.concatenate(residuals)
        return df


    # time series operators
    @staticmethod
    def _ts_return(df, price_col, period):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        df[f'ts_return_{period}'] = df.groupby(['date_stock'])[price_col].transform(
            lambda x: x.pct_change(periods=period)
        )
        return df

    @staticmethod
    def _ts_exponential_moving_average(df, cols, windows=[]):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_ema_{col}_{window}'] = df.groupby(['date_stock'])[col].transform(
                    lambda x: x.ewm(span=window, adjust=False).mean()
                )
        return df

    @staticmethod
    def _ts_std(df, cols, windows=[]):
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_std_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
        return df
    
    @staticmethod
    #具体来说，RSI 是基于过去14个交易日的价格变化来计算的
    def _ts_rsi(df, price_col, period=14):
        """计算相对强弱指数（RSI）"""
        delta = df[price_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        df[f'RSI_{price_col}_{period}'] = 100 - (100 / (1 + rs))
        return df

    @staticmethod
    def _ts_momentum(df, price_col, n):
        """计算动量（过去n天的变化）"""
        df[f'momentum_{price_col}_{n}'] = df[price_col].diff(periods=n)
        return df

    @staticmethod
    def _ts_ema(df, price_col, span):
        """计算指数移动平均（EMA）"""
        df[f'EMA_{price_col}_{span}'] = df[price_col].ewm(span=span, adjust=False).mean()
        return df

    @staticmethod
    def _ts_rank(df, cols):
        """计算排名"""
        for col in cols:
            df[f'rank_{col}'] = df[col].rank(pct=True)
        return df

    @staticmethod
    def _ts_quantile(df, cols, quantile=0.8):
        """计算80%分位数"""
        for col in cols:
            df[f'quantile_{col}_{quantile}'] = df.groupby('date_stock')[col].transform(lambda x: x.quantile(quantile))
        return df

    @staticmethod
    def _cs_cross_feature(df, cols):
        """生成交叉特征"""
        for c in itertools.combinations(cols, 2):
            df[f'cross_{c[0]}_{c[1]}'] = df[c[0]] * df[c[1]]
        return df
    def _ma(df, price_col, window):
        """计算移动平均线（MA）"""
        df[f'MA_{price_col}_{window}'] = df[price_col].rolling(window=window).mean()
        return df

    @staticmethod
    def _macd(df, price_col, short_window=12, long_window=26, signal_window=9):
        """计算移动平均收敛散布指标（MACD）"""
        df['MACD'] = df[price_col].ewm(span=short_window, adjust=False).mean() - df[price_col].ewm(span=long_window, adjust=False).mean()
        df['MACD_Signal'] = df['MACD'].ewm(span=signal_window, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        return df

    @staticmethod
    def _bollinger_bands(df, price_col, window=20, num_sd=2):
        """计算布林带（Bollinger Bands）"""
        df['MA'] = df[price_col].rolling(window=window).mean()
        df['Upper_Band'] = df['MA'] + (df[price_col].rolling(window=window).std() * num_sd)
        df['Lower_Band'] = df['MA'] - (df[price_col].rolling(window=window).std() * num_sd)
        return df

    @staticmethod
    def _vwap(df, price_col, volume_col):
        """计算成交量加权平均价格（VWAP）"""
        df['cum_price_volume'] = (df[price_col] * df[volume_col]).cumsum()
        df['cum_volume'] = df[volume_col].cumsum()
        df['VWAP'] = df['cum_price_volume'] / df['cum_volume']
        return df

    @staticmethod
    def _volatility(df, price_col, window=21):
        """计算波动率（Volatility）"""
        df[f'Volatility_{price_col}_{window}'] = df[price_col].rolling(window=window).std()
        return df