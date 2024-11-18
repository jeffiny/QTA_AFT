import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import itertools 
import statsmodels.api as sm

'''
Content: 没有check重复部分, water water enough
'''

class Operators:

    # 横截面幂次操作（例如平方、立方）
    @staticmethod
    def _cs_power(df, cols, power=2):
        for col in cols:
            df[f'cs_power_{col}_{power}'] = df[col] ** power
        return df

    # 横截面最大最小值之差
    @staticmethod
    def _cs_range(df, cols):
        df['cs_range'] = df[cols].max(axis=1) - df[cols].min(axis=1)
        return df

    # 横截面标准化 (Min-Max Normalization)
    @staticmethod
    def _cs_min_max_normalize(df, cols):
        for col in cols:
            min_val = df[col].min()
            max_val = df[col].max()
            df[f'cs_min_max_norm_{col}'] = (df[col] - min_val) / (max_val - min_val)
        return df

    # 横截面百分比变化
    @staticmethod
    def _cs_percentage_change(df, cols):
        for c in itertools.combinations(cols, 2):
            df[f'cs_pct_change_{c[0]}_{c[1]}'] = (df[c[1]] - df[c[0]]) / df[c[0]] * 100
        return df

    # 横截面加权平均
    @staticmethod
    def _cs_weighted_average(df, cols, weights):
        weighted_sum = sum(df[col] * weight for col, weight in zip(cols, weights))
        total_weight = sum(weights)
        df['cs_weighted_avg'] = weighted_sum / total_weight
        return df


    # time series operators
    # 累计和 (Cumulative Sum)
    @staticmethod
    def _ts_cumsum(df, col):
        df[f'ts_cumsum_{col}'] = df[col].cumsum()
        return df

    # 指数加权平均 (Exponentially Weighted Moving Average, EWMA)
    @staticmethod
    def _ts_ewm(df, col, span=20):
        df[f'ts_ewm_{col}_{span}'] = df[col].ewm(span=span, adjust=False).mean()
        return df

    # 时间序列差值 (Difference)
    @staticmethod
    def _ts_diff(df, col, periods=1):
        #take care periods set
        df[f'ts_diff_{col}_{periods}'] = df[col].diff(periods=periods)
        return df

    # 滑动最小值和最大值 (Rolling Min and Max)
    @staticmethod
    def _ts_moving_min_max(df, col, window=14):
        df[f'ts_moving_min_{col}_{window}'] = df[col].rolling(window=window).min()
        df[f'ts_moving_max_{col}_{window}'] = df[col].rolling(window=window).max()
        return df

    # 移动窗口范围 (Rolling Range)
    @staticmethod
    def _ts_rolling_range(df, col, window=14):
        min_series = df[col].rolling(window=window).min()
        max_series = df[col].rolling(window=window).max()
        df[f'ts_rolling_range_{col}_{window}'] = max_series - min_series
        return df
    
# 多元运算符
    @staticmethod
    def cs_correlation(df, col1, col2):
        df[f'cs_corr_{col1}_{col2}'] = df[[col1, col2]].corr().iloc[0, 1]
        return df
    
    # 按组操作：根据给定列分组并聚合目标列。
    @staticmethod
    def cs_group_aggregate(df, group_col, target_col, agg_func='mean'):
        df[f'cs_group_{agg_func}_{target_col}_{group_col}'] = df.groupby(group_col)[target_col].transform(agg_func)
        return df
    
    @staticmethod
    def cs_zscore(df, col):
        mean_val = df[col].mean()
        std_val = df[col].std()
        df[f'cs_zscore_{col}'] = (df[col] - mean_val) / std_val
        return df
    @staticmethod
    def cs_ols_beta(df, independent_col, dependent_col):
        # 横截面最小二乘回归beta值：计算自变量对因变量的beta值
        X = sm.add_constant(df[independent_col])
        y = df[dependent_col]
        model = sm.OLS(y, X).fit()
        df[f'cs_beta_{dependent_col}_{independent_col}'] = model.params[independent_col]
        return df

    def generate_binary_factors(df, cols, operator_func):
        # 遍历所有因子组合
        for col1, col2 in itertools.combinations(cols, 2):
            df = operator_func(df, col1, col2)
            # df = cs_correlation(df, col1, col2)
            # df = cs_group_aggregate(df, col1, col2)
        return df


