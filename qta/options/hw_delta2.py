import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from pandas import DateOffset
from sklearn.linear_model import LinearRegression


class CVaR:
    def __init__(self, q) -> None:
        self.q = q
        self.__name__ = f"{self.q}-CVaR"

    def __call__(self, x):
        if not isinstance(x, pd.Series):
            raise TypeError
        var = np.quantile(-x, self.q)
        return var + np.mean(np.maximum(-x - var, 0)) / (1 - self.q)


class VaR:
    def __init__(self, q) -> None:
        self.q = q
        self.__name__ = f"{self.q}-VaR"

    def __call__(self, x):
        if not isinstance(x, pd.Series):
            raise TypeError
        return np.quantile(-x, self.q)


def step_benchmark(
        action: np.ndarray, 
        stock_path: np.ndarray, 
        option_path: np.ndarray, 
        effc_on_path: np.ndarray, 
        trancost,
        optionid=None
    ):
    """
    Step the current path with benchmark delta. (In general, just used in test)
    Save the cumulative reward and the final wealth
    in self.reserve[f'tst_cum_{name}'] and self.reserve[f'tst_wealth_{name}'] respectively.
    If hope to save the action path, do it in running script used test dataframe. 
    """
    rate_path = effc_on_path[:-1]
    action_path = action[:-1]
    cum_rate = (1 + rate_path[::-1]).cumprod()[::-1]
    rate_initial_cash = (option_path[:-1] - action_path * stock_path[:-1] - trancost(action_path * stock_path[:-1])) * cum_rate
    delta_path = np.concatenate(([0], action_path))
    stock_flow = (action_path - delta_path[:-1]) * stock_path[:-1]
    rate_cash_flow = (stock_flow + trancost(stock_flow)) * cum_rate
    cum_rate_cash_flow = rate_cash_flow[::-1].cumsum()[::-1]
    cum_rate_cash_flow = np.concatenate((cum_rate_cash_flow[1:], [0]))
    hedging_err_path = rate_initial_cash - cum_rate_cash_flow + (action_path[-1] * stock_path[-1] - option_path[-1])
    hedging_err_path = np.concatenate((hedging_err_path, [0]))
    return hedging_err_path


class Data(object):
    def __init__(self, path='hw_delta2.csv') -> None:
        # Data:
        # Plz use the tau column as time-to-expiration rather than calculate it yourself, since it's a little complicated
        # stock_price: the S&P 500 index value
        # option_price: the midprice of option
        # Other variables are same as those in OptionMetrics, or not relevant
        # the data is from 2013-08-31 to 2015-08-31, you can use shorter window to do regression
        self.df = pd.read_csv(path)
        self.df['date'] = pd.to_datetime(self.df['date'].str.strip(), errors='coerce')
        #self.df['date'] = pd.to_datetime(self.df['date'])
        self.df['exdate'] = pd.to_datetime(self.df['exdate'])    

        # self.join_lvf()
        # self.join_pl()


    def join_lvf(self):
        cols = {'lvf_x0', 'lvf_x1', 'lvf_smile_slope', 'delta_lvf'}
        self.df = self.df.drop(cols.intersection(self.df.columns), axis=1)
            
        df_grouped =  self.df.groupby(['date', 'exdate', 'cp_flag', 'am_settlement'])

        def calculate_lvf(single_df):
            lvf_data = single_df[['optionid', 'date', 'impl_volatility', 'strike_price', 'volume', 'delta']].dropna()
            lvf_data = lvf_data[lvf_data['impl_volatility'] != 0]
            lvf_data = lvf_data[lvf_data['volume'] > 0.1]
            lvf_data = lvf_data[(lvf_data['delta'].abs() >= 0.05) & (lvf_data['delta'].abs() <= 0.95)]
            
            if len(lvf_data) <= 4:
                x0, x1 = 0, 0
            else:
                y = lvf_data['impl_volatility']
                x = lvf_data['strike_price']
                fit_coef = np.polyfit(x, y, 2)
                x0, x1 = fit_coef[0], fit_coef[1]
            
            single_df['lvf_x0'] = x0
            single_df['lvf_x1'] = x1
                
            return single_df[['optionid', 'date', 'lvf_x0', 'lvf_x1']]

        # Use joblib to acceleration
        lvf_iter = Parallel(n_jobs=12)(delayed(calculate_lvf)(single_df) for idx, single_df in df_grouped)
        df_add = pd.concat(lvf_iter, axis=0)
        self.df = pd.merge(self.df, df_add, how = 'left', on = ['date', 'optionid'])
        self.df['lvf_smile_slope'] = 2 * self.df['lvf_x0'] * self.df['strike_price'] + self.df['lvf_x1']
        lvf_delta = self.df['lvf_smile_slope'] * self.df['vega'] + self.df['delta']

        # lvf_delta.loc[(df['delta'].abs() >= 0.05) & (df['delta'].abs() <= 0.95)] = df.loc[(df['delta'].abs() >= 0.05) & (df['delta'].abs() <= 0.95), 'delta']
        change_sign = lvf_delta * self.df['delta'] < 0
        lvf_delta.loc[change_sign] = 0
        # lvf_delta.loc[change_sign] = df.loc[change_sign, 'delta']
        self.df[f'delta_lvf'] = lvf_delta

    def join_pl(self):
        """
        You can use this function to test your delta.
        This function will add columns like final_PL_{method}_{cons/prop}, 
        the final P&L using delta_{method} as delta to hedge option from now to expiration given no/proportional transaction costs.

        Or you can use the metric in the HW paper.
        """
        cons_tc = lambda x: 0.
        prop_tc = lambda x: np.abs(x) * 0.001
        def map_df(single_df: pd.DataFrame):
            for col in single_df.columns:
                if 'delta' in col[:5]:
                    col_name = 'final_PL_bs' if col == 'delta' else col.replace('delta', 'final_PL')
                    if len(single_df) == 1:
                        single_df[col_name + '_cons'] = 0
                        single_df[col_name + '_prop'] = 0
                        continue
                    single_df[col_name + '_cons'] = step_benchmark(
                        single_df[col].values, single_df['stock_price'].values, 
                        single_df['option_price'].values, single_df['effective_on'].values, 
                        cons_tc
                    )
                    single_df[col_name + '_prop'] = step_benchmark(
                        single_df[col].values, single_df['stock_price'].values, 
                        single_df['option_price'].values, single_df['effective_on'].values, 
                        prop_tc
                    )
            return single_df

        df_grouped = self.df.groupby('optionid')    
        df_iter = Parallel(n_jobs=12)(delayed(map_df)(single_df) for idx, single_df in df_grouped)
        #df_iter = delayed(map_df)(single_df) for idx, single_df in df_grouped
        self.df = pd.concat(df_iter, axis=0)

    def join_hw(self, **kwargs):
        """
        Plz calculate HW delta here and add a column like "delta_hw" to self.df
        you may add arguments like regression window length, data selection criteria
        """
        # the normalized X delta seems to be just return
        # 修正为差分+使用t+1天和t天的diff拟合第t天，后面拟合频率修改为月频率因此只会用到每个月第一天的数据
        days_diff = -1
        """self.df['option_diff'] = self.df.groupby('optionid')['option_price'].diff(days_diff)
        self.df['stock_diff'] = self.df.groupby('optionid')['stock_price'].diff(days_diff)
        self.df['Vega_over_sqrt_T'] = self.df['vega'] / np.sqrt(self.df['tau'])
        self.df['Delta_S_over_S'] = self.df['option_diff'] / self.df['stock_price']
        self.df['X'] = self.df['Vega_over_sqrt_T'] * self.df['Delta_S_over_S']
        self.df['X_delta_BS'] = self.df['X'] * self.df['delta']
        self.df['X_delta_BS_squared'] = self.df['X'] * (self.df['delta'] ** 2)
        #同除第t天S
        self.df['y'] = (self.df['option_diff'] - self.df['delta'] * self.df['stock_diff']) / self.df['stock_price']
        df_regression = self.df.dropna(subset=['y', 'X', 'X_delta_BS', 'X_delta_BS_squared'])
        df_calls = df_regression[df_regression['cp_flag'] == 'C']
        df_puts = df_regression[df_regression['cp_flag'] == 'P']"""

        option_diff = self.df.groupby('optionid')['option_price'].diff(days_diff) / self.df['stock_price']
        self.df['stock_diff'] = self.df.groupby('optionid')['stock_price'].diff(days_diff) / self.df['stock_price']
        self.df['x_0_hw'] = self.df['vega'] / np.sqrt(self.df['tau']) / self.df['stock_price']
        self.df['x_1_hw'] = self.df['x_0_hw'] * self.df['delta']
        self.df['x_2_hw'] = self.df['x_1_hw'] * self.df['delta']
        self.df['y_hw'] = option_diff - self.df['delta'] * self.df['stock_diff']
        df_regression = self.df.dropna(subset=['y_hw', 'x_0_hw', 'x_1_hw', 'x_2_hw', 'stock_diff'])
        df_calls = df_regression[df_regression['cp_flag'] == 'C']
        df_puts = df_regression[df_regression['cp_flag'] == 'P']

        # 月频：每个月用前36个月拟合一组a，b，c， 效果和下面注释掉的日频的基本一样
        # def rolling_regression(data, window_size_months=36):
        #      # 将日期转换为月份
        #      data['year_month'] = data['date'].dt.to_period('M')
        #      # 获取唯一的月份列表
        #      unique_months = data['year_month'].unique()
        #      data['a'] = np.nan
        #      data['b'] = np.nan
        #      data['c'] = np.nan
        #      results = []
        #      for current_month in unique_months:
        #          print(current_month)
        #          # 定义回归窗口的起止日期
        #          end_date = (pd.Period(current_month, freq='M') - 1).end_time
        #          start_date = (pd.Period(current_month, freq='M') - window_size_months).start_time
        #          # 选择窗口内的数据
        #          window_data = data[(data['date'] >= start_date) & (data['date'] <= end_date)]
        #          #修改判断条件为筛选到的日期超过3个月的交易日60天
        #          if (window_data.date.max() - window_data.date.min()).days > 60:
        #              #len(window_data) > 3:  # 确保有样本（需要定多少样本量的阈值？）
        #              # X = window_data[['X', 'X_delta_BS', 'X_delta_BS_squared']]
        #              # y = window_data['y']
        #              X = window_data[['stock_diff']].values * window_data[['x_0_hw', 'x_1_hw', 'x_2_hw']].values
        #              y= window_data['y_hw']
        #              model = LinearRegression(fit_intercept=False)
        #              model.fit(X, y)
        #              a_hat, b_hat, c_hat = model.coef_
        #              # 保存结果
        #              results.append({
        #                  'date': end_date,
        #                  'a_hat': model.coef_[0],
        #                  'b_hat': model.coef_[1],
        #                  'c_hat': model.coef_[2],
        #                  'intercept': model.intercept_,
        #                  'n_samples': len(window_data)
        #              })
        #              current_month_data_index = data[(data['year_month'] == current_month)].index
        #              data.loc[current_month_data_index, 'a'] = a_hat
        #              data.loc[current_month_data_index, 'b'] = b_hat
        #              data.loc[current_month_data_index, 'c'] = c_hat
        #      return pd.DataFrame(results)
         #每天用前36个月拟合一组a，b，c
        def rolling_regression(data, window_size_months=36):
           # 获取唯一的日期列表
            unique_date = sorted(data['date'].unique())
            data['a'] = np.nan
            data['b'] = np.nan
            data['c'] = np.nan
            results = []
            for current_date in unique_date:
                # 定义回归窗口的起止日期
                end_date = current_date
                start_date = pd.to_datetime(current_date) - DateOffset(months=window_size_months)
                # 选择窗口内的数据
                # print(start_date,":",end_date)
                # Will there be information leakage if including the current date??
                window_data = data[(data['date'] >= start_date) & (data['date'] < end_date)]
                if (window_data.date.max() - window_data.date.min()).days>60:#len(window_data) >= 3:  # 确保有样本（需要定多少样本量的阈值？）
                    X = window_data[['stock_diff']].values * window_data[['x_0_hw', 'x_1_hw', 'x_2_hw']].values
                    y = window_data['y_hw']
                    model = LinearRegression(fit_intercept=False)
                    model.fit(X, y)
                    a_hat, b_hat, c_hat = model.coef_
                    # 保存结果
                    results.append({
                        'date': end_date,
                        'a_hat': model.coef_[0],
                        'b_hat': model.coef_[1],
                        'c_hat': model.coef_[2],
                        'intercept': model.intercept_,
                        'n_samples': len(window_data)
                    })
                    current_date_index = data[(data['date'] == current_date)].index
                    data.loc[current_date_index, 'a'] = a_hat
                    data.loc[current_date_index, 'b'] = b_hat
                    data.loc[current_date_index, 'c'] = c_hat
            return pd.DataFrame(results)

        results_calls = rolling_regression(df_calls)
        results_puts = rolling_regression(df_puts)
        import matplotlib.pyplot as plt
        for c in ['a', 'b', 'c']:
            plt.plot(results_calls.index, results_calls[f'{c}_hat'])
        plt.show()
        for c in ['a', 'b', 'c']:
            plt.plot(results_calls.index, results_puts[f'{c}_hat'])
        plt.show()

        self.df['delta_hw'] = np.nan
        #print(self.df.loc[df_calls.index,'delta_hw'])
        """self.df.loc[df_calls.index,'delta_hw'] = df_calls["delta"] + df_calls['a'] * df_calls['X'] \
            + df_calls['b'] * df_calls['X_delta_BS'] \
            + df_calls['c'] * df_calls['X_delta_BS_squared'] 
        self.df.loc[df_puts.index,'delta_hw'] = df_puts["delta"] + df_puts['a'] * df_puts['X'] \
            + df_puts['b'] * df_puts['X_delta_BS'] \
            + df_puts['c'] * df_puts['X_delta_BS_squared'] """
        #print(self.df.loc[df_calls.index,'delta_hw'].isnull().sum())
        self.df.loc[df_calls.index,'delta_hw'] = df_calls["delta"] \
            + df_calls['a'] * df_calls['x_0_hw'] \
            + df_calls['b'] * df_calls['x_1_hw'] \
            + df_calls['c'] * df_calls['x_2_hw'] 
        self.df.loc[df_puts.index,'delta_hw'] = df_puts["delta"] \
            + df_puts['a'] * df_puts['x_0_hw'] \
            + df_puts['b'] * df_puts['x_1_hw'] \
            + df_puts['c'] * df_puts['x_2_hw'] 
    def result_print(self):
        self.df['option_diff'] = self.df.groupby('optionid')['option_price'].diff(-1) / self.df['stock_price']
        self.df['stock_diff'] = self.df.groupby('optionid')['stock_price'].diff(-1) / self.df['stock_price']
        self.df['delta_group'] = round(self.df['delta'] * 10) / 10
        SSE_MV = self.df.groupby(["delta_group", "cp_flag"]).apply(
            lambda x: (((x['option_diff'] - x["delta_hw"] * (x['stock_diff'])) ** 2).sum())).reset_index()
        SSE_BS = self.df.groupby(["delta_group", "cp_flag"]).apply(
            lambda x: (((x['option_diff'] - x["delta"] * (x['stock_diff'])) ** 2).sum())).reset_index()
        SSE = SSE_MV.merge(SSE_BS, on=['delta_group', 'cp_flag'])
        SSE.columns = ['delta_group', 'cp_flag'] + ["SSE_MV", "SSE_BS"]
        SSE["GAIN"] = 1 - SSE["SSE_MV"] / SSE["SSE_BS"]
        return SSE

if __name__ == '__main__':
    data = Data()
    # data.join_lvf()
    data.join_hw()
    data.join_pl()
    pl_cols = [c for c in data.df.columns if 'final_PL' in c]
    print(data.df[pl_cols].dropna().agg([
        'mean', 'std', VaR(0.95), CVaR(0.95), VaR(0.975), CVaR(0.975)
    ]))
    print(data.result_print())

