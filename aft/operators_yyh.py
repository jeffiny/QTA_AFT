import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
import pandas as pd
import numpy as np
import numpy as np

'''
没有检查和其他同学的重合内容
'''


class Operators:

    @staticmethod
    def _ts_log_return(df, price_col, period=1):
        """计算对数收益率"""
        df[f'Log_Return_{price_col}_{period}'] = np.log(df[price_col] / df[price_col].shift(period))
        return df

    @staticmethod
    def _ts_normal_return(df, price_col, period=1):
        """计算普通收益率"""
        df[f'Normal_Return_{price_col}_{period}'] = (df[price_col] / df[price_col].shift(period) - 1)
        return df

    @staticmethod
    def _ts_lag_feature(df, col, periods=1):
        """创建滞后特征"""
        df[f'Lag_{col}_{periods}'] = df[col].shift(periods)
        return df

    @staticmethod
    def _ts_diff_feature(df, col):
        """计算一阶差分特征"""
        df[f'Diff_{col}'] = df[col].diff()
        return df

    @staticmethod
    def _ts_square_feature(df, col):
        """计算平方特征"""
        df[f'Square_{col}'] = df[col] ** 2
        return df

    @staticmethod
    def _ts_cube_feature(df, col):
        """计算立方特征"""
        df[f'Cube_{col}'] = df[col] ** 3
        return df

    @staticmethod
    def _ts_abs_diff_feature(df, col1, col2):
        """计算两列绝对差特征"""
        df[f'Abs_Diff_{col1}_{col2}'] = (df[col1] - df[col2]).abs()
        return df

    @staticmethod
    def _ts_min_max_scaler(df, col):
        """最小最大标准化特征"""
        df[f'MinMaxScaler_{col}'] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df

    @staticmethod
    def _ts_log_transform(df, col):
        """对数变换特征"""
        df[f'Log_{col}'] = np.log(df[col] + 1)  # 加1防止对0取对数
        return df

    @staticmethod
    def _ts_z_score_scaler(df, col):
        """Z分数标准化特征"""
        mean = df[col].mean()
        std = df[col].std()
        df[f'Z_Score_{col}'] = (df[col] - mean) / std
        return df

    @staticmethod
    def _ts_simple_moving_average(df, price_col, window):
        """计算简单移动平均（SMA）"""
        df[f'SMA_{price_col}_{window}'] = df[price_col].rolling(window=window).mean()
        return df

    @staticmethod
    def _ts_weighted_moving_average(df, price_col, window):
        """计算加权移动平均（WMA）"""
        weights = np.arange(1, window + 1)
        weights = weights / weights.sum()
        df[f'WMA_{price_col}_{window}'] = (df[price_col].rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True))
        return df

    @staticmethod
    def _ts_median_price(df, high_col, low_col, window):
        """计算中位数价格"""
        df[f'Median_Price_{high_col}_{low_col}'] = df[[high_col, low_col]].median(axis=1).rolling(window=window).mean()
        return df

    @staticmethod
    def _ts_rate_of_change(df, price_col, period=1):
        """计算价格变动率（ROC）"""
        df[f'ROC_{price_col}_{period}'] = df[price_col].pct_change(periods=period)
        return df

    @staticmethod
    def _ts_true_strength_index(df, price_col, r1=25, r2=50):
        """计算真实强度指数（TSI）"""
        df[f'TSI_{price_col}_{r1}_{r2}'] = df[price_col].diff(12).ewm(span=r1, adjust=False).mean() - df[price_col].diff(12).ewm(span=r2, adjust=False).mean()
        return df

    @staticmethod
    def _ts_on_balance_volume(df, price_col, volume_col):
        """计算成交量平衡（OBV）"""
        df['OBV'] = np.where(df[price_col] > df[price_col].shift(1), df[volume_col], 0)
        df['OBV'] = np.where(df[price_col] < df[price_col].shift(1), df['OBV'] - df[volume_col], df['OBV'])
        df['OBV'] = df['OBV'].cumsum()
        return df

    @staticmethod
    def _ts_force_index(df, price_col, volume_col, period=13):
        """计算力度指数"""
        df[f'Force_Index_{price_col}_{volume_col}_{period}'] = (df[price_col].diff() * df[volume_col].diff()).rolling(window=period).sum()
        return df
 