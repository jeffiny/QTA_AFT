import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from itertools import combinations

'''
Editor: Yiming Jiang
Time: 2024/11/2
Content: 新增绝对值、截面排名、截面最小最大值及均值偏离量
         注释了重合部分
'''

class Operators:
    # @staticmethod
    # def _add_(df, cols):
    #     for tup in combinations(cols, 2):
    #         df[f"{tup[0]}_{tup[1]}_add"] = df[tup[0]] + df[tup[1]]
    #     return df

    # @staticmethod
    # def _minus_(df, cols):
    #     for tup in combinations(cols, 2):
    #         df[f"{tup[0]}_{tup[1]}_minus"] = df[tup[0]] - df[tup[1]]
    #     return df

    # @staticmethod
    # def _mul_(df, cols):
    #     for tup in combinations(cols, 2):
    #         df[f"{tup[0]}_{tup[1]}_mul"] = df[tup[0]] * df[tup[1]]
    #     return df

    # @staticmethod
    # def _div_(df, cols):
    #     for tup in combinations(cols, 2):
    #         df[f"{tup[0]}_{tup[1]}_div"] = df[tup[0]] / df[tup[1]]
    #     return df

    # @staticmethod
    # def _max_(df, cols):
    #     for tup in combinations(cols, 2):
    #         df[f"{tup[0]}_{tup[1]}_max"] = df[[tup[0], tup[1]]].max(axis=1)
    #     return df

    # @staticmethod
    # def _min_(df, cols):
    #     for tup in combinations(cols, 2):
    #         df[f"{tup[0]}_{tup[1]}_min"] = df[[tup[0], tup[1]]].min(axis=1)
    #     return df

    # @staticmethod
    # def _mean_(df, cols):
    #     for tup in combinations(cols, 2):
    #         df[f"{tup[0]}_{tup[1]}_min"] = df[[tup[0], tup[1]]].mean(axis=1)
    #     return df

    @staticmethod
    def _abs_(df, cols):
        return df[cols].abs()

    @staticmethod
    def _cs_rank_(df, cols):
        """逐时分组计算排名"""
        df_group = df.groupby(["date_id", "seconds_in_bucket"])
        for col in cols:
            df[f"{col}_rank"] = df_group[col].rank(method="min")
        return df

    @staticmethod
    def _deviation_from_mean_(df, cols):
        df_group = df.groupby(["date_id", "seconds_in_bucket"])
        for col in cols:
            df[f"{col}_rank"] = df[col] - df_group[col].transform("mean")
        return df

    @staticmethod
    def _deviation_from_max_(df, cols):
        df_group = df.groupby(["date_id", "seconds_in_bucket"])
        for col in cols:
            df[f"{col}_rank"] = df[col] - df_group[col].transform("max")
        return df

    @staticmethod
    def _deviation_from_min_(df, cols):
        df_group = df.groupby(["date_id", "seconds_in_bucket"])
        for col in cols:
            df[f"{col}_rank"] = df[col] - df_group[col].transform("min")
        return df

