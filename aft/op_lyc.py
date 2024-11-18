import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
from scipy.signal import hilbert
from scipy.stats import entropy
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fftpack import fft
from pywt import wavedec

'''
Just Use o1 to supplement zhifan's code

交叉截面算子

算子名称                            描述
_cs_add                         对列的两两组合进行加法
_cs_zscore_add                          对列进行截面 z-score 标准化后，两两相加
_cs_rank_add                            对列进行截面排名后，两两相加
_cs_multiply                            对列的两两组合进行乘法
_cs_divide                          对列的两两组合进行除法
_cs_subtract                            对列的两两组合进行减法
_cs_pairwise_min                            对列的两两组合取最小值
_cs_pairwise_max                            对列的两两组合取最大值
_cs_abs                         对列取绝对值
_cs_square                          对列进行平方
_cs_sqrt                            对列取平方根
_cs_log                         对列取对数（需确保值为正数）
_cs_exp                         对列取指数
_cs_pow                         对列取指定幂次
_cs_modulo                          对列取模
_cs_mean                            计算列在当前截面上的均值
_cs_std                         计算列在当前截面上的标准差
_cs_median                          计算列在当前截面上的中位数
_cs_min                         计算列在当前截面上的最小值
_cs_max                         计算列在当前截面上的最大值
_cs_quantile                            计算列在当前截面上的分位数
_cs_winsorize                           对列进行去极值处理
_cs_outlier_filter                          对列进行异常值过滤
_cs_quantile_transform                          对列进行分位数变换
_cs_rank_multiply                           对列进行截面排名后，两两相乘
_cs_rank_divide                         对列进行截面排名后，两两相除
_cs_indicator                           根据阈值对列生成指标函数特征。
_cs_interaction_terms                           生成列的所有交互项
_cs_polynomial_features                         对列生成多项式特征

时间序列算子

算子名称                            描述
_ts_mean                            计算列的滚动均值
_ts_std                         计算列的滚动标准差
_ts_corr                            计算列之间的滚动相关系数
_ts_zscore                          计算列的滚动 z-score 标准化
_ts_rank                            计算列的滚动排名
_ts_slope                           计算列的滚动回归斜率
_ts_diff                            计算列的滚动差分
_ts_min                         计算列的滚动最小值
_ts_max                         计算列的滚动最大值
_ts_median                          计算列的滚动中位数
_ts_sum                         计算列的滚动求和
_ts_skew                            计算列的滚动偏度
_ts_kurt                            计算列的滚动峰度
_ts_quantile                            计算列的滚动分位数
_ts_product                         计算列的滚动乘积
_ts_shift                           对列进行滞后处理
_ts_cumsum                          计算列的累积和
_ts_cumprod                         计算列的累积乘积
_ts_abs                         对列取绝对值
_ts_log                         对列取对数（需确保值为正数）
_ts_exp                         对列取指数
_ts_ewm_mean                            计算列的指数加权移动平均
_ts_ewm_std                         计算列的指数加权移动标准差
_ts_ewm_corr                            计算列之间的指数加权移动相关系数
_ts_cov                         计算列之间的滚动协方差
_ts_autocorr                            计算列的滚动自相关系数
_ts_cumreturn                           计算列的累积收益率
_ts_momentum                            计算列的滚动动量
_ts_return                          计算列的滚动收益率
_ts_decay_linear                            计算列的线性衰减加权平均
_ts_residual                            提取列与目标列做滚动回归后的残差
_ts_detrend                         对列去除滚动趋势
_ts_seasonal_decompose                          对列进行季节性分解，提取趋势、季节性和残差成分。
_ts_hilbert_transform                           对列进行 Hilbert 变换，提取瞬时幅值、相位和频率。
_ts_entropy                         计算列的滚动熵
_ts_fourier_transform                           对列进行快速傅里叶变换，提取频域特征。
_ts_wavelet_transform                           对列进行小波变换，提取多尺度特征。
_ts_hurst_exponent                          计算列的滚动 Hurst 指数
_ts_fractal_dimension                           计算列的滚动分形维数
_ts_spectral_density                            计算列的滚动功率谱密度
_ts_statistical_test                            对列进行滚动统计检验（如 ADF 检验）
_ts_peak_detection                          对列进行峰值检测
_ts_zero_crossing_rate                          计算列的过零率
_ts_pca                         对列进行主成分分析，提取主要成分
_ts_kernel_density                          对列进行核密度估计
_ts_dynamic_time_warping                            计算列之间的动态时间规整距离


'''

class Operators:
    # 交叉截面算子
    @staticmethod
    def _cs_add(df, cols):
        # 对cols中的两两列进行加法组合
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_add_{c[0]}_{c[1]}': lambda x: x[c[0]] + x[c[1]]})
        return df

    @staticmethod
    def _cs_zscore_add(df, cols):
        # 对cols中的每一列进行截面z-score标准化，然后两两相加
        for col in cols:
            df[f"cs_zscore_{col}"] = (df[col] - df.groupby(['time_id'])[col].transform('mean')) / df.groupby(['time_id'])[col].transform('std')
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_add_{c[0]}_{c[1]}': lambda x: x['cs_zscore_'+c[0]] + x['cs_zscore_'+c[1]]})
        return df

    @staticmethod
    def _cs_rank_add(df, cols):
        # 对cols中的每一列进行截面排序（百分比），然后两两相加
        for col in cols:
            df[f"cs_rank_{col}"] = df.groupby(['time_id'])[col].rank(pct=True)
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_add_{c[0]}_{c[1]}': lambda x: x['cs_rank_'+c[0]] + x['cs_rank_'+c[1]]})
        return df

    @staticmethod
    def _cs_multiply(df, cols):
        # 对cols中的两两列进行乘法组合
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_multiply_{c[0]}_{c[1]}': lambda x: x[c[0]] * x[c[1]]})
        return df

    @staticmethod
    def _cs_divide(df, cols):
        # 对cols中的两两列进行除法组合
        for c in itertools.combinations(cols, 2):
            assert (df[c[1]] != 0).all()
            df = df.assign(**{f'cs_divide_{c[0]}_{c[1]}': lambda x: x[c[0]] / x[c[1]]})
        return df

    @staticmethod
    def _cs_subtract(df, cols):
        # 对cols中的两两列进行减法组合
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_subtract_{c[0]}_{c[1]}': lambda x: x[c[0]] - x[c[1]]})
        return df

    @staticmethod
    def _cs_pairwise_min(df, cols):
        # 对cols中的两两列取最小值
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_pairwise_min_{c[0]}_{c[1]}': lambda x: x[[c[0], c[1]]].min(axis=1)})
        return df

    @staticmethod
    def _cs_pairwise_max(df, cols):
        # 对cols中的两两列取最大值
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_pairwise_max_{c[0]}_{c[1]}': lambda x: x[[c[0], c[1]]].max(axis=1)})
        return df

    @staticmethod
    def _cs_abs(df, cols):
        # 对cols中的每一列取绝对值
        for col in cols:
            df[f'cs_abs_{col}'] = df[col].abs()
        return df

    @staticmethod
    def _cs_square(df, cols):
        # 对cols中的每一列取平方
        for col in cols:
            df[f'cs_square_{col}'] = df[col] ** 2
        return df

    @staticmethod
    def _cs_sqrt(df, cols):
        # 对cols中的每一列取平方根
        for col in cols:
            df[f'cs_sqrt_{col}'] = np.sqrt(df[col])
        return df

    @staticmethod
    def _cs_log(df, cols):
        # 对cols中的每一列取对数，需确保值为正数
        for col in cols:
            df[f'cs_log_{col}'] = np.log(df[col].replace(0, np.nan))
        return df

    @staticmethod
    def _cs_exp(df, cols):
        # 对cols中的每一列取指数
        for col in cols:
            df[f'cs_exp_{col}'] = np.exp(df[col])
        return df

    @staticmethod
    def _cs_pow(df, cols, powers):
        # 对cols中的每一列取不同的幂次
        for col in cols:
            for power in powers:
                df[f'cs_pow_{col}_{power}'] = df[col] ** power
        return df

    @staticmethod
    def _cs_modulo(df, cols, mod_value):
        # 对cols中的每一列取模
        for col in cols:
            df[f'cs_modulo_{col}_{mod_value}'] = df[col] % mod_value
        return df

    @staticmethod
    def _cs_mean(df, cols):
        # 计算cols中的每一列在当前截面上的均值
        for col in cols:
            df[f'cs_mean_{col}'] = df.groupby('time_id')[col].transform('mean')
        return df

    @staticmethod
    def _cs_std(df, cols):
        # 计算cols中的每一列在当前截面上的标准差
        for col in cols:
            df[f'cs_std_{col}'] = df.groupby('time_id')[col].transform('std')
        return df

    @staticmethod
    def _cs_median(df, cols):
        # 计算cols中的每一列在当前截面上的中位数
        for col in cols:
            df[f'cs_median_{col}'] = df.groupby('time_id')[col].transform('median')
        return df

    @staticmethod
    def _cs_min(df, cols):
        # 计算cols中的每一列在当前截面上的最小值
        for col in cols:
            df[f'cs_min_{col}'] = df.groupby('time_id')[col].transform('min')
        return df

    @staticmethod
    def _cs_max(df, cols):
        # 计算cols中的每一列在当前截面上的最大值
        for col in cols:
            df[f'cs_max_{col}'] = df.groupby('time_id')[col].transform('max')
        return df

    @staticmethod
    def _cs_quantile(df, cols, quantiles=[]):
        # 计算cols中的每一列在当前截面上的分位数
        for col in cols:
            for quantile in quantiles:
                df[f'cs_quantile_{col}_{quantile}'] = df.groupby('time_id')[col].transform(lambda x: x.quantile(quantile))
        return df
    
    @staticmethod
    def _cs_quantile_transform(df, cols, n_quantiles=100):
        # 对cols中的每一列进行分位数变换
        for col in cols:
            df[f'cs_quantile_transform_{col}'] = pd.qcut(
                df[col], q=n_quantiles, labels=False, duplicates='drop') / (n_quantiles - 1)
        return df

    @staticmethod
    def _cs_rank_multiply(df, cols):
        # 对cols中的每一列进行截面排名，然后两两相乘
        for col in cols:
            df[f"cs_rank_{col}"] = df.groupby(['time_id'])[col].rank(pct=True)
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_rank_multiply_{c[0]}_{c[1]}': lambda x: x[f'cs_rank_{c[0]}'] * x[f'cs_rank_{c[1]}']})
        return df

    @staticmethod
    def _cs_rank_divide(df, cols):
        # 对cols中的每一列进行截面排名，然后两两相除
        for col in cols:
            df[f"cs_rank_{col}"] = df.groupby(['time_id'])[col].rank(pct=True)
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_rank_divide_{c[0]}_{c[1]}': lambda x: x[f'cs_rank_{c[0]}'] / x[f'cs_rank_{c[1]}']})
        return df

    @staticmethod
    def _cs_indicator(df, cols, thresholds):
        # 根据阈值对cols中的每一列生成指标函数特征
        for col, threshold in zip(cols, thresholds):
            df[f'cs_indicator_{col}'] = (df[col] > threshold).astype(int)
        return df

    @staticmethod
    def _cs_interaction_terms(df, cols):
        # 生成cols中所有列的交互项
        for c in itertools.combinations(cols, 2):
            df = df.assign(**{f'cs_interaction_{c[0]}_{c[1]}': lambda x: x[c[0]] * x[c[1]]})
        return df

    @staticmethod
    def _cs_polynomial_features(df, cols, degree=2):
        # 对cols中的每一列生成多项式特征
        for col in cols:
            for d in range(2, degree + 1):
                df[f'cs_poly_{col}_{d}'] = df[col] ** d
        return df

    # 时间序列算子
    @staticmethod
    def _ts_mean(df, cols, windows=[]):
        # 对cols中的每一列计算滚动均值
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_mean_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).mean())
        return df

    # 时间序列算子
    @staticmethod
    def _ts_mean(df, cols, windows=[]):
        # 对cols中的每一列计算滚动均值
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_mean_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
        return df

    @staticmethod
    def _ts_std(df, cols, windows=[]):
        # 对cols中的每一列计算滚动标准差
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_std_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
        return df

    @staticmethod
    def _ts_corr(df, cols, windows=[]):  # 速度较慢
        # 对cols中的两两列计算滚动相关系数
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for c in itertools.combinations(cols, 2):
            for window in windows:
                df[f"ts_corr_{c[0]}_{c[1]}_{window}"] = df.groupby(['date_stock']).apply(
                    lambda x: x[[c[0], c[1]]].rolling(window=window, min_periods=1).corr().iloc[0::2, 1].reset_index(level=1, drop=True)
                ).reset_index(level=0, drop=True)
        return df

    @staticmethod
    def _ts_zscore(df, cols, windows=[]):
        # 对cols中的每一列计算滚动z-score标准化
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_zscore_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(
                    lambda x: (x - x.rolling(window=window, min_periods=1).mean()) / x.rolling(window=window, min_periods=1).std())
        return df

    @staticmethod
    def _ts_rank(df, cols, windows=[]):
        # 对cols中的每一列计算滚动排名
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_rank_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).rank(pct=True))
        return df

    @staticmethod
    def _ts_slope(df, cols, windows=[]):  # 速度较慢
        # 对cols中的每一列计算滚动回归斜率
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
        # 对cols中的每一列计算滚动差分
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f"ts_diff_{col}_{window}"] = df.groupby(['date_stock'])[col].transform(lambda x: x.diff(periods=window))
        return df

    @staticmethod
    def _ts_min(df, cols, windows=[]):
        # 对cols中的每一列计算滚动最小值
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_min_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).min())
        return df

    @staticmethod
    def _ts_max(df, cols, windows=[]):
        # 对cols中的每一列计算滚动最大值
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_max_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).max())
        return df

    @staticmethod
    def _ts_median(df, cols, windows=[]):
        # 对cols中的每一列计算滚动中位数
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_median_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).median())
        return df

    @staticmethod
    def _ts_sum(df, cols, windows=[]):
        # 对cols中的每一列计算滚动求和
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_sum_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).sum())
        return df

    @staticmethod
    def _ts_skew(df, cols, windows=[]):
        # 对cols中的每一列计算滚动偏度
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_skew_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).skew())
        return df

    @staticmethod
    def _ts_kurt(df, cols, windows=[]):
        # 对cols中的每一列计算滚动峰度
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_kurt_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).kurt())
        return df

    @staticmethod
    def _ts_quantile(df, cols, windows=[], quantiles=[]):
        # 对cols中的每一列计算滚动分位数
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                for quantile in quantiles:
                    df[f'ts_quantile_{col}_{window}_{quantile}'] = df.groupby('date_stock')[col].transform(
                        lambda x: x.rolling(window=window, min_periods=1).quantile(quantile))
        return df

    @staticmethod
    def _ts_product(df, cols, windows=[]):
        # 对cols中的每一列计算滚动乘积
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_product_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=1).apply(np.prod, raw=True))
        return df

    @staticmethod
    def _ts_shift(df, cols, periods=[]):
        # 对cols中的每一列进行滞后处理
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for period in periods:
                df[f'ts_shift_{col}_{period}'] = df.groupby('date_stock')[col].shift(periods=period)
        return df

    @staticmethod
    def _ts_cumsum(df, cols):
        # 对cols中的每一列计算累积和
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            df[f'ts_cumsum_{col}'] = df.groupby('date_stock')[col].cumsum()
        return df

    @staticmethod
    def _ts_cumprod(df, cols):
        # 对cols中的每一列计算累积乘积
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            df[f'ts_cumprod_{col}'] = df.groupby('date_stock')[col].cumprod()
        return df

    @staticmethod
    def _ts_abs(df, cols):
        # 对cols中的每一列取绝对值
        for col in cols:
            df[f'ts_abs_{col}'] = df[col].abs()
        return df

    @staticmethod
    def _ts_log(df, cols):
        # 对cols中的每一列取对数，需确保值为正数
        for col in cols:
            df[f'ts_log_{col}'] = np.log(df[col].replace(0, np.nan))
        return df

    @staticmethod
    def _ts_exp(df, cols):
        # 对cols中的每一列取指数
        for col in cols:
            df[f'ts_exp_{col}'] = np.exp(df[col])
        return df

    @staticmethod
    def _ts_ewm_mean(df, cols, spans=[]):
        # 对cols中的每一列计算指数加权移动平均
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for span in spans:
                df[f'ts_ewm_mean_{col}_{span}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.ewm(span=span, min_periods=1).mean())
        return df

    @staticmethod
    def _ts_ewm_std(df, cols, spans=[]):
        # 对cols中的每一列计算指数加权移动标准差
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for span in spans:
                df[f'ts_ewm_std_{col}_{span}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.ewm(span=span, min_periods=1).std())
        return df

    @staticmethod
    def _ts_ewm_corr(df, cols, spans=[]):
        # 对cols中的两两列计算指数加权移动相关系数
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for c in itertools.combinations(cols, 2):
            for span in spans:
                df[f'ts_ewm_corr_{c[0]}_{c[1]}_{span}'] = df.groupby('date_stock').apply(
                    lambda x: x[[c[0], c[1]]].ewm(span=span, min_periods=1).corr().iloc[0::2, 1].reset_index(level=1, drop=True)
                ).reset_index(level=0, drop=True)
        return df

    @staticmethod
    def _ts_cov(df, cols, windows=[]):
        # 对cols中的两两列计算滚动协方差
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for c in itertools.combinations(cols, 2):
            for window in windows:
                df[f'ts_cov_{c[0]}_{c[1]}_{window}'] = df.groupby('date_stock').apply(
                    lambda x: x[[c[0], c[1]]].rolling(window=window, min_periods=1).cov().iloc[0::2, 1].reset_index(level=1, drop=True)
                ).reset_index(level=0, drop=True)
        return df

    @staticmethod
    def _ts_autocorr(df, cols, windows=[]):
        # 对cols中的每一列计算滚动自相关系数
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_autocorr_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window).apply(lambda y: y.autocorr(lag=1), raw=False)
                )
        return df

    @staticmethod
    def _ts_cumreturn(df, cols):
        # 对cols中的每一列计算累积收益率
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            df[f'ts_cumreturn_{col}'] = df.groupby('date_stock')[col].apply(lambda x: (1 + x).cumprod() - 1)
        return df

    @staticmethod
    def _ts_momentum(df, cols, windows=[]):
        # 对cols中的每一列计算滚动动量
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_momentum_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.diff(window))
        return df

    @staticmethod
    def _ts_return(df, cols, windows=[]):
        # 对cols中的每一列计算滚动收益率
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_return_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.pct_change(periods=window))
        return df

    @staticmethod
    def _ts_decay_linear(df, cols, windows=[]):
        # 对cols中的每一列计算线性衰减加权平均
        def get_decay_weights(window):
            weights = np.arange(1, window + 1)
            weights = weights / weights.sum()
            return weights

        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                weights = get_decay_weights(window)
                df[f'ts_decay_linear_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window).apply(lambda y: np.dot(y, weights) if len(y) == window else np.nan, raw=True)
                )
        return df
    
    @staticmethod
    def _ts_residual(df, cols, target_col, windows=[]):
        # 对cols中的每一列与目标列做滚动回归，提取残差
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                def rolling_residual(x):
                    if len(x) < 2:
                        return np.nan
                    y = x[target_col]
                    X = sm.add_constant(x[col])
                    model = sm.OLS(y, X)
                    results = model.fit()
                    return results.resid.iloc[-1]
                df[f'ts_residual_{col}_{window}'] = df.groupby('date_stock').apply(
                    lambda x: x[[col, target_col]].rolling(window=window, min_periods=2).apply(
                        rolling_residual, raw=False))
        return df

    @staticmethod
    def _ts_detrend(df, cols, windows=[]):
        # 对cols中的每一列去除滚动趋势
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                def detrend(y):
                    x = np.arange(len(y))
                    A = np.vstack([x, np.ones(len(y))]).T
                    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
                    return y - (m * x + c)
                df[f'ts_detrend_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        lambda y: detrend(y)[-1], raw=False))
        return df

    @staticmethod
    def _ts_seasonal_decompose(df, cols, model='additive', period=12):
        # 对cols中的每一列进行季节性分解，提取趋势、季节性和残差成分
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            def decompose(x):
                if len(x) < period * 2:
                    return [np.nan, np.nan, np.nan]
                result = seasonal_decompose(x, model=model, period=period, extrapolate_trend='freq')
                return [result.trend.iloc[-1], result.seasonal.iloc[-1], result.resid.iloc[-1]]
            df[[f'ts_trend_{col}', f'ts_seasonal_{col}', f'ts_resid_{col}']] = df.groupby('date_stock')[col].transform(
                lambda x: pd.DataFrame([decompose(x)], columns=[f'ts_trend_{col}', f'ts_seasonal_{col}', f'ts_resid_{col}']))
        return df

    @staticmethod
    def _ts_hilbert_transform(df, cols):
        # 对cols中的每一列进行Hilbert变换，提取瞬时幅值、相位和频率
        for col in cols:
            analytic_signal = hilbert(df[col])
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))
            instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * 1)
            df[f'ts_hilbert_amplitude_{col}'] = amplitude_envelope
            df[f'ts_hilbert_phase_{col}'] = instantaneous_phase
            df[f'ts_hilbert_frequency_{col}'] = np.append(instantaneous_frequency, np.nan)
        return df

    @staticmethod
    def _ts_entropy(df, cols, windows=[]):
        # 对cols中的每一列计算滚动熵
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                def rolling_entropy(x):
                    counts = np.histogram(x, bins=10)[0] + 1  # 加1以避免零概率
                    return entropy(counts)
                df[f'ts_entropy_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        rolling_entropy, raw=False))
        return df

    @staticmethod
    def _ts_fourier_transform(df, cols):
        # 对cols中的每一列进行快速傅里叶变换，提取频域特征
        for col in cols:
            fft_values = fft(df[col].fillna(0))
            df[f'ts_fft_real_{col}'] = np.real(fft_values)
            df[f'ts_fft_imag_{col}'] = np.imag(fft_values)
        return df

    @staticmethod
    def _ts_wavelet_transform(df, cols, wavelet='db1', level=2):
        # 对cols中的每一列进行小波变换，提取多尺度特征
        for col in cols:
            coeffs = wavedec(df[col].fillna(0), wavelet=wavelet, level=level)
            for i, coeff in enumerate(coeffs):
                df[f'ts_wavelet_{col}_coeff_{i}'] = coeff
        return df

    @staticmethod
    def _ts_hurst_exponent(df, cols, windows=[]):
        # 对cols中的每一列计算滚动Hurst指数
        def hurst(ts):
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]*2.0

        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_hurst_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        lambda y: hurst(y), raw=False))
        return df

    @staticmethod
    def _ts_fractal_dimension(df, cols, windows=[]):
        # 对cols中的每一列计算滚动分形维数
        def fractal_dimension(y):
            N = len(y)
            L = []
            for k in range(1, int(N/2)):
                Lk = 0
                for i in range(0, N - k, k):
                    Lk += abs(y[i + k] - y[i])
                Lk = (Lk * (N - 1)) / (k * (N - k))
                L.append(Lk)
            coeffs = np.polyfit(np.log(range(1, len(L)+1)), np.log(L), 1)
            return coeffs[0]

        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                df[f'ts_fractal_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        lambda y: fractal_dimension(y), raw=False))
        return df

    @staticmethod
    def _ts_spectral_density(df, cols, windows=[]):
        # 对cols中的每一列计算滚动功率谱密度
        from scipy.signal import periodogram
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                def spectral_density(x):
                    f, Pxx_den = periodogram(x)
                    return np.sum(Pxx_den)
                df[f'ts_spectral_density_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        spectral_density, raw=False))
        return df

    @staticmethod
    def _ts_statistical_test(df, cols, windows=[]):
        # 对cols中的每一列进行滚动统计检验，例如ADF检验
        from statsmodels.tsa.stattools import adfuller
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                def adf_test(x):
                    result = adfuller(x, autolag='AIC')
                    return result[0]  # 返回ADF统计量
                df[f'ts_adf_stat_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        adf_test, raw=False))
        return df

    @staticmethod
    def _ts_peak_detection(df, cols, windows=[]):
        # 对cols中的每一列进行峰值检测
        from scipy.signal import find_peaks
        for col in cols:
            df[f'ts_peaks_{col}'] = df[col].apply(lambda x: len(find_peaks(x)[0]))
        return df

    @staticmethod
    def _ts_zero_crossing_rate(df, cols, windows=[]):
        # 对cols中的每一列计算过零率
        df['date_stock'] = df['date_id'].astype(str) + '_' + df['stock_id'].astype(str)
        for col in cols:
            for window in windows:
                def zero_crossing_rate(x):
                    return ((x[:-1] * x[1:]) < 0).sum() / len(x)
                df[f'ts_zero_crossing_{col}_{window}'] = df.groupby('date_stock')[col].transform(
                    lambda x: x.rolling(window=window, min_periods=2).apply(
                        zero_crossing_rate, raw=False))
        return df

    @staticmethod
    def _ts_pca(df, cols, n_components=2):
        # 对cols中的列进行主成分分析，提取主要成分
        from sklearn.decomposition import PCA
        pca = PCA(n_components=n_components)
        principal_components = pca.fit_transform(df[cols].fillna(0))
        for i in range(n_components):
            df[f'ts_pca_component_{i+1}'] = principal_components[:, i]
        return df

    @staticmethod
    def _ts_kernel_density(df, cols):
        # 对cols中的每一列进行核密度估计
        from sklearn.neighbors import KernelDensity
        for col in cols:
            kde = KernelDensity(kernel='gaussian', bandwidth=0.75).fit(df[[col]].dropna())
            log_density = kde.score_samples(df[[col]].fillna(0))
            df[f'ts_kde_{col}'] = np.exp(log_density)
        return df

    @staticmethod
    def _ts_dynamic_time_warping(df, cols):
        # 对cols中的两两列计算动态时间规整距离
        from dtaidistance import dtw
        for c in itertools.combinations(cols, 2):
            distance = dtw.distance(df[c[0]].fillna(0), df[c[1]].fillna(0))
            df[f'ts_dtw_{c[0]}_{c[1]}'] = distance
        return df