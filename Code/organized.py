import json
import os
import random
import numpy as np
import pandas as pd
from statsforecast import StatsForecast

from utilsforecast.losses import smape
from statsforecast.models import MSTL, AutoARIMA
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
plt.rcParams['lines.linewidth'] = 1.5
dark_style = {
    'figure.facecolor': '#212946',
    'axes.facecolor': '#212946',
    'savefig.facecolor': '#212946',
    'axes.grid': True,
    'axes.grid.which': 'both',
    'axes.spines.left': False,
    'axes.spines.right': False,
    'axes.spines.top': False,
    'axes.spines.bottom': False,
    'grid.color': '#2A3459',
    'grid.linewidth': '1',
    'text.color': '0.9',
    'axes.labelcolor': '0.9',
    'xtick.color': '0.9',
    'ytick.color': '0.9',
    'font.size': 12,
}
plt.rcParams.update(dark_style)
plt.rcParams['figure.figsize'] = (15, 10)
pd.options.display.float_format = '{:.2f}'.format

HORIZON = 48

POWER = 10  # MW
CAPACITY = 20  # MWh
MAX_TRADE = POWER / 2  # MW per half-hour
EFFICIENCY = 1

# 'stochastic', 'robust' or 'deterministic'
MODEL_TYPE = 'robust'
RISK_AVERSE_FACTOR = 0
BETA = 0.99

# 'block', 'standard', 'monte_carlo' or 'dexter'
SCENARIO_TYPE = 'dexter'
NUM_SCENARIOS = 100

SHOW_FINAL_SCHEDULE = False
PLOT_CVARS = False
SHOW_SCENARIOS = False


def main():
    ems = EMS(Data())
    max_days = ems.data.testing_data['DA'].shape[0] // 48

    results = ems.run(max_days)
    results['params'] = {
        'POWER': POWER,
        'CAPACITY': CAPACITY,
        'MAX_TRADE': MAX_TRADE,
        'EFFICIENCY': EFFICIENCY,
        'RISK_AVERSE_FACTOR': RISK_AVERSE_FACTOR,
        'BETA': BETA,
        'NUM_SCENARIOS': NUM_SCENARIOS,
        'SCENARIO_TYPE': SCENARIO_TYPE,
    }
    print(
        f"Overall profit: {sum([day['profit'] for day in results['results'].values()])}")

    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    subfolder_name = f'{MODEL_TYPE}/battery_{POWER}MW_{CAPACITY}MWh'
    subfolder_path = os.path.join(results_dir, subfolder_name)
    if not os.path.exists(subfolder_path):
        os.makedirs(subfolder_path)
    subfolder_path = os.path.join(results_dir, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)
    if MODEL_TYPE == 'stochastic':
        results_file = os.path.join(
            subfolder_path, f'run_results_l={RISK_AVERSE_FACTOR}.json')
    else:
        results_file = os.path.join(subfolder_path, 'run_results.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)


class Data:
    def __init__(self):
        df = pd.read_excel('./Data/IDA & DA Ierland 2023.xlsx')
        self.da = pd.DataFrame({
            'ds': df['Datetime'],
            'unique_id': 1,
            'y': df['IE DA EUR']
        })
        self.ida1 = pd.DataFrame({
            'ds': df['Datetime'],
            'unique_id': 1,
            'y': df['IE IDA1 EUR price']
        })
        self.ida2 = pd.DataFrame({
            'ds': df['Datetime'],
            'unique_id': 1,
            'y': df['IE IDA2 EUR price']
        })
        self.cross_vals = {}
        self.cross_vals['DA'] = pd.read_excel(
            './Data/Crossvalidation/cross_val_30_days_da.xlsx')
        self.cross_vals['IDA1'] = pd.read_excel(
            './Data/Crossvalidation/cross_val_30_days_ida1.xlsx')
        self.cross_vals['IDA2'] = pd.read_excel(
            './Data/Crossvalidation/cross_val_30_days_ida2.xlsx')
        self.__split_data()

    def __split_data(self):
        train_ida1, test_ida1 = self.ida1[self.ida1['ds'].dt.month <
                                          12], self.ida1[self.ida1['ds'].dt.month >= 12]
        train_ida2, test_ida2 = self.ida2[self.ida2['ds'].dt.month <
                                          12], self.ida2[self.ida2['ds'].dt.month >= 12]
        train_da, test_da = self.da[self.da['ds'].dt.month <
                                    12], self.da[self.da['ds'].dt.month >= 12]
        self.date = test_da['ds'].iloc[0].date()
        self.training_data = {'DA': train_da,
                              'IDA1': train_ida1, 'IDA2': train_ida2}
        self.testing_data = {'DA': test_da,
                             'IDA1': test_ida1, 'IDA2': test_ida2}
        self.prediction_day = pd.DataFrame({
            'ds': pd.date_range(start=self.date, periods=48, freq='30min'),
            'unique_id': 1,
        })

    def get_forecast_from_file(self, auction_name):
        exo_vars = self.training_data[auction_name].drop(
            columns=['y', 'ds', 'unique_id']).columns.tolist()

        print(
            f"Retrieving: ./Data/Forecasts/{auction_name}/{self.date}_exo{exo_vars}.xlsx")

        forecast = pd.read_excel(
            f'./Data/Forecasts/{auction_name}/{self.date}_exo{exo_vars}.xlsx')

        return forecast['MSTL'].values

    def move_to_next_day(self):
        self.__update_past_errors()
        self.date += pd.Timedelta(days=1)
        self.training_data['DA'] = self.da[self.da['ds'].dt.date <
                                           self.date]
        self.training_data['IDA1'] = self.ida1[self.ida1['ds'].dt.date <
                                               self.date]
        self.training_data['IDA2'] = self.ida2[self.ida2['ds'].dt.date <
                                               self.date]
        self.testing_data['DA'] = self.da[self.da['ds'].dt.date >=
                                          self.date]
        self.testing_data['IDA1'] = self.ida1[self.ida1['ds'].dt.date >=
                                              self.date]
        self.testing_data['IDA2'] = self.ida2[self.ida2['ds'].dt.date >=
                                              self.date]
        self.prediction_day = pd.DataFrame({
            'ds': pd.date_range(start=self.date, periods=48, freq='30min'),
            'unique_id': 1,
        })

    def __update_past_errors(self):
        for auction_name in self.cross_vals.keys():
            print(f"\nUpdating cross-validation for {auction_name}")
            cross_val = self.cross_vals[auction_name]
            y = self.get_actual_prices(auction_name)
            mstl = self.get_forecast_from_file(auction_name)
            trading_hours = cross_val['ds'].dt.hour.unique()
            timeframe_mask = self.prediction_day['ds'].dt.floor(
                'h').dt.hour.isin(trading_hours)
            df = pd.DataFrame({
                'ds': self.prediction_day['ds'].loc[timeframe_mask],
                'y': y,
                'MSTL': mstl,
                'error': y - mstl
            })
            self.cross_vals[auction_name] = pd.concat(
                [cross_val, df]).reset_index(drop=True)
            print(
                f"Forecasting errors taken from past {len(cross_val['ds'].dt.date.unique())} days")
            print("Mean error updated from", cross_val['error'].mean(
            ), "to", self.cross_vals[auction_name]['error'].mean())
            print("Std error updated from", cross_val['error'].std(
            ), "to", self.cross_vals[auction_name]['error'].std())

    def realize_prices(self, auction_name):
        date_mask = self.testing_data[auction_name]['ds'].dt.date == self.date
        auction_prices = self.testing_data[auction_name].loc[date_mask, 'y'].values
        self.prediction_day[f'y_{auction_name}'] = auction_prices

        # if auction_name == 'DA':
        auctions = ['DA', 'IDA1', 'IDA2']
        for key in auctions[auctions.index(auction_name)+1:]:
            print(f"Adding {auction_name} prices to {key} training data")
            self.training_data[key] = self.training_data[key].copy()
            merged_data = pd.merge(self.training_data[key],
                                   self.training_data[auction_name][[
                                       'ds', 'y']],
                                   on='ds',
                                   how='left',  # Keep all original rows for key
                                   suffixes=('', f'_{auction_name}'))
            self.training_data[key] = merged_data

    def get_actual_prices(self, auction_name):
        date_mask = self.testing_data[auction_name]['ds'].dt.date == self.date
        return self.testing_data[auction_name].loc[date_mask, 'y'].dropna().values

    def plot_scenarios(self, scenarios, auction_name):
        date_mask = self.testing_data[auction_name]['ds'].dt.date == self.date
        real_prices = self.testing_data[auction_name].loc[date_mask, [
            'y', 'ds']].dropna()
        plt.plot(real_prices['ds'], real_prices['y'], label='Real Prices')
        for scenario in scenarios:
            plt.plot(real_prices['ds'], scenario)
        plt.legend()
        plt.show()

    def calculate_smape(self, auction_name, forecast):
        date_mask = self.testing_data[auction_name]['ds'].dt.date == self.date
        actual = self.testing_data[auction_name].loc[date_mask, 'y'].dropna(
        ).values
        nmae = np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual))
        # return round(smape(pd.DataFrame({'y': actual, 'y_hat': forecast, 'unique_id': 1}), [
        # 'y_hat']).iloc[0, 1]*100, 2)
        return nmae*100


class EMS:
    def __init__(self, data):
        self.data = data

    def __forecast(self, auction_name):
        try:
            forecast = self.data.get_forecast_from_file(auction_name)
            print("SMAPE:", self.data.calculate_smape(auction_name, forecast))
            return forecast
        except:
            train = self.data.training_data[auction_name]
            train.dropna(inplace=True)
            train.reset_index(drop=True, inplace=True)
            exos = train.drop(
                columns=['y', 'ds', 'unique_id']).columns.tolist()
            print(
                f"FILE NOT FOUND: Forecasting for {auction_name} on {self.data.date} using exogeneous variables: {exos}")
            trading_hours = train['ds'].dt.hour.unique()

            X_df = self.data.prediction_day[train.drop(columns=['y']).columns]
            timeframe_mask = X_df['ds'].dt.floor(
                'h').dt.hour.isin(trading_hours)
            X_df = X_df.loc[timeframe_mask]

            trading_length = len(trading_hours) * 2

            print("timeframe:", X_df['ds'].iloc[0], "to", X_df['ds'].iloc[-1])
            model = self.__create_mstl(
                seasons=[trading_length, 2*trading_length])
            forecast = model.forecast(
                df=train, h=len(X_df), X_df=X_df)

            forecast.reset_index(drop=True, inplace=True)
            forecast['ds'] = X_df['ds'].reset_index(drop=True)

            print("SMAPE:", self.data.calculate_smape(
                auction_name, forecast['MSTL'].values))
            forecast.to_excel(
                f'./Data/Forecasts/{auction_name}/{self.data.date}_exo{exos}.xlsx', index=False)
            return forecast['MSTL'].values

    def __create_mstl(self, seasons):
        models = [
            MSTL(season_length=seasons,
                 trend_forecaster=AutoARIMA(seasonal=True)
                 )
        ]
        sf = StatsForecast(
            models=models,
            freq='30min',
        )
        return sf

    def get_scenarios(self, forecast, auction_name, type):
        if type == 'block':
            return self.__get_block_scenarios(forecast, auction_name)
        elif type == 'standard':
            return self.__get_standard_scenarios(forecast, auction_name)
        elif type == 'monte_carlo':
            return self.__get_monte_carlo_scenarios(forecast, auction_name)
        elif type == 'dexter':
            return self.__get_dexter_scenarios(forecast, auction_name)
        else:
            raise ValueError("Invalid scenario type")

    def __get_dexter_scenarios(self, forecast, auction_name):
        cross_val = self.data.cross_vals[auction_name].copy()
        residuals = cross_val.groupby(
            [cross_val['ds'].dt.hour, cross_val['ds'].dt.minute])['error']

        quantiles = np.linspace(0, 1, NUM_SCENARIOS)
        scenarios = [forecast]
        for quantile in quantiles:
            errors = residuals.quantile(quantile).values
            scenario = np.add(forecast, errors)
            scenarios.append(scenario)

        if SHOW_SCENARIOS:
            self.data.plot_scenarios(scenarios, auction_name)

        return scenarios

    def __get_block_scenarios(self, forecast, auction_name, block_size=8):
        cross_val = self.data.cross_vals[auction_name].copy()
        # split each day into blocks of errors of size 'block_size'
        blocks_by_day = [
            group['error'].values.reshape(-1, block_size) for _, group in cross_val.groupby(cross_val['ds'].dt.date)]
        # group blocks by time of day (gather blocks from all days with the same time of day)
        grouped_blocks = list(zip(*blocks_by_day))

        scenarios = []
        for _ in range(NUM_SCENARIOS):
            scenario = []
            # for each time of day, select a random block of errors and append it to the forecast
            for block_index, block in enumerate(grouped_blocks):
                correlated_block = random.choice(block)
                prediction_block = forecast[block_index * block_size:(
                    block_index + 1) * block_size]
                scenario.extend(prediction_block + correlated_block)
            scenarios.append(scenario)

        scenarios.append(forecast)
        scenarios = np.array(scenarios)
        if SHOW_SCENARIOS:
            self.data.plot_scenarios(scenarios, auction_name)
        return scenarios

    def __get_standard_scenarios(self, predictions, auction_name):
        cross_val = self.data.cross_vals[auction_name]

        grouped = cross_val.groupby(
            [cross_val['ds'].dt.hour, cross_val['ds'].dt.minute])
        error_distr = grouped['error'].agg(['mean', 'std'])

        percentiles = [
            *np.linspace(0, 5, (NUM_SCENARIOS//2)),
            # *np.linspace(5, 100, 10),
            *np.linspace(95, 100, (NUM_SCENARIOS//2))
        ]
        sampled_errors = np.random.normal(error_distr['mean'].values[:, None],
                                          error_distr['std'].values[:, None],
                                          (len(error_distr), 10000))
        scenario_errors = np.percentile(sampled_errors, percentiles, axis=1)
        scenarios = predictions[:, None] + scenario_errors.T

        # y_peaks = cross_val.groupby(cross_val['ds'].dt.day)[
        #     'y'].apply(list).apply(lambda row: np.argmax(row))
        # mstl_peaks = cross_val.groupby(cross_val['ds'].dt.day)[
        #     'MSTL'].apply(list).apply(lambda row: np.argmax(row))
        # peaks_errors = mstl_peaks - y_peaks
        # shift_value = int(np.floor(peaks_errors.mean()))
        # if shift_value != 0:
        #     print("Shift value:", shift_value)
        #     shifted_scenarios = np.roll(scenarios, shift=-shift_value, axis=0)
        #     scenarios = np.hstack([scenarios, shifted_scenarios])

        scenarios = np.hstack([scenarios, predictions[:, None]])
        scenarios_df = pd.DataFrame(scenarios)
        res = scenarios_df.values.T.tolist()

        if SHOW_SCENARIOS:
            self.data.plot_scenarios(res, auction_name)

        return res

    def __get_monte_carlo_scenarios(self, predictions, auction_name):
        cross_val = self.data.cross_vals[auction_name]

        grouped = cross_val.groupby(
            [cross_val['ds'].dt.hour, cross_val['ds'].dt.minute])
        error_distr = grouped['error'].agg(['mean', 'std'])

        sampled_errors = np.random.normal(
            loc=error_distr['mean'].values[:, None],
            scale=error_distr['std'].values[:, None],
            size=(len(error_distr), NUM_SCENARIOS)
        )

        predictions_repeated = np.repeat(
            predictions[:, None], NUM_SCENARIOS, axis=1)

        scenarios = predictions_repeated + \
            sampled_errors

        scenarios_df = pd.DataFrame(scenarios.T)
        res = scenarios_df.values.tolist()

        if SHOW_SCENARIOS:
            self.data.plot_scenarios(res, auction_name)

        return res

    def __get_trading_window(self, auction_name):
        start_time = self.data.training_data[auction_name].dropna()[
            'ds'].dt.hour.min() * 2
        end_time = (self.data.training_data[auction_name].dropna()[
            'ds'].dt.hour.max() + 1) * 2
        return start_time, end_time

    def __calculate_profit(self, schedules):
        profit = 0
        for auction in ['DA', 'IDA1', 'IDA2']:
            if auction in schedules:
                date_mask = self.data.testing_data[auction]['ds'].dt.date == self.data.date
                actual_prices = self.data.testing_data[auction].loc[date_mask, 'y'].fillna(0
                                                                                           ).values
                profit += np.dot(-np.array(actual_prices),
                                 schedules[auction])
        return profit

    def __show_schedule(self, schedules, prices, soc_levels):
        _, (price_ax, power_ax) = plt.subplots(2, 1)

        schedule = pd.DataFrame()
        for auction, bids in schedules.items():
            schedule[auction] = bids
        schedule.index = self.data.prediction_day['ds']
        schedule.plot(kind='bar', stacked=True, ax=power_ax)

        power_ax.plot(soc_levels, color='yellow', label='SOC')

        for auction, price_path in zip(schedule.keys(), prices):
            price_ax.plot(price_path, label=f'{auction} Prices')
        price_ax.set_ylabel("Price (eur)")
        power_ax.sharex(price_ax)

        plt.suptitle(f"Battery: {POWER} MW, {CAPACITY} MWh")
        plt.xticks(range(0, 48, 8), range(0, 24, 4))
        plt.xlabel("Time (hours)")
        power_ax.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        price_ax.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        power_ax.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        price_ax.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        price_ax.legend()
        power_ax.legend()
        plt.show()

    def __plot_cvars(self, first_stage_forecast, second_stage_scenarios,
                     start_time_y, end_time_y, soc):
        res = {}
        for l in np.linspace(0, 1, 11):
            bids, adjustments, _, cvar, e_scenarios_profit = self.__solve_two_stage(
                first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, soc, risk_averse_factor=l)

            scenario_profits = [np.dot(-np.array(first_stage_forecast), bids) + np.dot(-np.array(
                prices), trades) for trades, prices in zip(adjustments, second_stage_scenarios)]

            res[l] = {
                "cvar": cvar,
                "expected": np.dot(-np.array(first_stage_forecast), bids) + e_scenarios_profit,
                "worst": min(scenario_profits),
            }
        risk_averse_factors = list(res.keys())
        cvars = [res[l]["cvar"] for l in risk_averse_factors]
        expected = [res[l]["expected"] for l in risk_averse_factors]
        worst = [res[l]["worst"] for l in risk_averse_factors]
        plt.plot(risk_averse_factors, cvars, label="CVaR")
        plt.plot(risk_averse_factors, expected, label="Expected Profit")
        # plt.plot(risk_averse_factors, worst, label="Worst-Case Profit")
        plt.title(f"Battery: {POWER} MW, {CAPACITY} MWh")
        plt.xlabel("Risk Averse Factor")
        plt.ylabel("Value (eur)")
        plt.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        plt.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        plt.legend()
        plt.show()

    def __solve_two_stage_robust(self, x_forecast, y_scenarios,
                                 start_time_y, end_time_y, prev_bids):

        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)  # Turn off Gurobi output

        x = model.addVars(HORIZON, lb=-MAX_TRADE,
                          ub=MAX_TRADE, name="bids")
        y = model.addVars(HORIZON, NUM_SCENARIOS, lb=-MAX_TRADE, ub=MAX_TRADE,
                          name="adjustments")
        soc = model.addVars(HORIZON+1, NUM_SCENARIOS,
                            lb=0, ub=CAPACITY, name="soc")

        for s in range(NUM_SCENARIOS):
            model.addConstr(soc[0, s] == 0, name=f"initial_soc_s{s}")

        for t in range(HORIZON):
            for s in range(NUM_SCENARIOS):
                if t < start_time_y or t >= end_time_y:
                    model.addConstr(y[t, s] == 0)

                model.addConstr(
                    soc[t+1, s] == soc[t, s] + x[t] + y[t, s] + prev_bids[t],
                    name=f"soc_balance_t{t}_s{s}"
                )
                model.addConstr(
                    soc[t+1, s] <= soc[t, s] + POWER,
                    name=f"physical_charge_limit_t{t}_s{s}"
                )
                model.addConstr(
                    soc[t+1, s] >= soc[t, s] - POWER,
                    name=f"physical_discharge_limit_t{t}_s{s}"
                )

        scenario_profit = {s: gp.quicksum(
            -y_scenarios[s][t - start_time_y] * y[t, s] for t in range(start_time_y, end_time_y)) for s in range(NUM_SCENARIOS)}

        worst_y = model.addVar(name="worst-case_scenario_profit")
        model.addConstrs(
            worst_y <= scenario_profit[s] for s in range(NUM_SCENARIOS))

        profit_x = gp.quicksum(-x_forecast[t] * x[t] for t in range(HORIZON))
        model.setObjective(profit_x + worst_y,
                           GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            print("Forecasted profit x:", profit_x.getValue())
            print("Worst-case scenario profit:", worst_y.X)
            return (
                [x[t].x for t in range(HORIZON)],
                [[y[t, s].x for t in range(start_time_y, end_time_y)]
                 for s in range(NUM_SCENARIOS)],
                [[soc[t+1, s].x for t in range(HORIZON)]
                 for s in range(NUM_SCENARIOS)],
                np.nan,
                worst_y.X
            )
        else:
            raise Exception("Master is infeasible")

    def __solve_two_stage(self, x_forecast, y_scenarios,
                          start_time_y, end_time_y, prev_bids,
                          risk_averse_factor=RISK_AVERSE_FACTOR):

        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)  # Turn off Gurobi output
        # print("trading length y:", np.array(y_scenarios).shape[1])
        # print("start time y:", start_time_y)
        # print("end time y:", end_time_y)

        x = model.addVars(HORIZON, lb=-MAX_TRADE,
                          ub=MAX_TRADE, name="bids")
        y = model.addVars(HORIZON, NUM_SCENARIOS, lb=-MAX_TRADE, ub=MAX_TRADE,
                          name="adjustments")
        soc = model.addVars(HORIZON+1, NUM_SCENARIOS,
                            lb=0, ub=CAPACITY, name="soc")

        for s in range(NUM_SCENARIOS):
            model.addConstr(soc[0, s] == 0, name=f"initial_soc_s{s}")

        for t in range(HORIZON):
            for s in range(NUM_SCENARIOS):
                if t < start_time_y or t >= end_time_y:
                    model.addConstr(y[t, s] == 0)

                model.addConstr(
                    soc[t+1, s] == soc[t, s] + x[t] + y[t, s] + prev_bids[t],
                    name=f"soc_balance_t{t}_s{s}"
                )
                model.addConstr(
                    soc[t+1, s] <= soc[t, s] + POWER,
                    name=f"physical_charge_limit_t{t}_s{s}"
                )
                model.addConstr(
                    soc[t+1, s] >= soc[t, s] - POWER,
                    name=f"physical_discharge_limit_t{t}_s{s}"
                )

        scenario_profit = {s: gp.quicksum(
            -y_scenarios[s][t - start_time_y] * y[t, s] for t in range(start_time_y, end_time_y)) for s in range(NUM_SCENARIOS)}
        expected_scenario_profit = (
            1/NUM_SCENARIOS)*gp.quicksum(scenario_profit[s] for s in range(NUM_SCENARIOS))

        alpha = model.addVar(lb=0, name="VaR")
        z = model.addVars(NUM_SCENARIOS, lb=0, name="excess_loss")
        loss = model.addVars(NUM_SCENARIOS, lb=0, name="loss")
        cvar = model.addVar(lb=0, name="cvar")

        for s in range(NUM_SCENARIOS):
            model.addConstr(
                loss[s] >= expected_scenario_profit - scenario_profit[s])
            model.addConstr(z[s] >= loss[s] - alpha)
        model.addConstr(cvar >= alpha + (1 / (NUM_SCENARIOS * (1 - BETA)))
                        * gp.quicksum(z[s] for s in range(NUM_SCENARIOS)))

        expected_profit = gp.quicksum(
            (-x_forecast[t] * x[t]) for t in range(HORIZON)) + expected_scenario_profit
        model.setObjective((expected_profit) - (cvar *
                           risk_averse_factor), GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return (
                [x[t].x for t in range(HORIZON)],
                [[y[t, s].x for t in range(start_time_y, end_time_y)]
                 for s in range(NUM_SCENARIOS)],
                [[soc[t+1, s].x for t in range(HORIZON)]
                 for s in range(NUM_SCENARIOS)],
                cvar.X,
                expected_scenario_profit.getValue()
            )
        else:
            raise Exception("Master is infeasible")

    def __solve_single(self, forecast, prev_bids, start_time_x, end_time_x):
        print("Trading period length:", len(forecast))

        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)  # Turn off Gurobi output

        x = model.addVars(HORIZON, lb=-MAX_TRADE,
                          ub=MAX_TRADE, name="bids")
        soc = model.addVars(HORIZON+1, lb=0, ub=CAPACITY, name="soc")

        model.addConstr(soc[0] == 0, name="initial_soc")

        for t in range(HORIZON):
            if t < start_time_x or t >= end_time_x:
                model.addConstr(x[t] == 0)

            model.addConstr(
                soc[t+1] == soc[t] + x[t] + prev_bids[t],
                name=f"soc_balance_t{t}"
            )
            model.addConstr(
                soc[t+1] <= soc[t] + POWER,
                name=f"physical_charge_limit_t{t}"
            )
            model.addConstr(
                soc[t+1] >= soc[t] - POWER,
                name=f"physical_discharge_limit_t{t}"
            )

        profit = gp.quicksum(-forecast[t - start_time_x] * x[t]
                             for t in range(start_time_x, end_time_x))

        model.setObjective(profit, GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            schedule = [x[t].x for t in range(
                start_time_x, end_time_x)]
            profits = [-price*bid for price,
                       bid in zip(forecast, schedule)]
            if sum(profits) != model.objVal:
                print("Model's objective value:", model.objVal)
                print("Actual Profit:", sum(profits))
                raise Exception("Objective value does not match actual profit")
            if sum(schedule) > 0 and all([price > 0 for price in forecast]) and NUM_SCENARIOS == 0:
                print("schedule =", schedule)
                print("forecast =", forecast)
                raise Exception("Net volume left")
            if sum([x[t].x for t in range(HORIZON)]) > 0 and sum(schedule) == 0:
                print("somethings not right")
                print("schedule =", schedule)
                print("bids =", [x[t].x for t in range(HORIZON)])
                raise Exception("Schedules don't match")
            return (
                [x[t].x for t in range(HORIZON)],
                [soc[t+1].x for t in range(HORIZON)]
            )
        else:
            print(f"forecast = {forecast}")
            print(f"prev_bids = {prev_bids}")
            print(f"start_time = {start_time_x}")
            print(f"end_time = {end_time_x}")
            self.__show_schedule({'Previous Bids': prev_bids}, [
                                 forecast], np.cumsum(prev_bids))
            raise Exception("Master is infeasible")

    def __run_stochastic(self):
        print("Forecasting day:", self.data.date)

        first_stage = "DA"
        second_stage = "IDA1"
        next_stages = ["IDA2"]
        schedules = {}
        prev_bids = [0] * HORIZON
        stats = {}
        while second_stage:
            print("First stage:", first_stage)
            print("Second stage:", second_stage)

            first_stage_forecast = self.__forecast(first_stage)
            second_stage_scenarios = self.get_scenarios(
                self.__forecast(second_stage), second_stage, SCENARIO_TYPE)

            start_time_y, end_time_y = self.__get_trading_window(second_stage)

            print("BEST SCENARIO SMAPE:", min([
                self.data.calculate_smape(second_stage, second_stage_scenario)
                for second_stage_scenario in second_stage_scenarios
            ]))

            if PLOT_CVARS and first_stage == "DA" and self.data.date.day == 12:
                self.__plot_cvars(first_stage_forecast, second_stage_scenarios, start_time_y,
                                  end_time_y, prev_bids)

            bids, ys, _, cvar, _ = self.__solve_two_stage(
                first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, prev_bids)
            schedules[first_stage] = bids
            prev_bids = np.add(prev_bids, bids)

            first_stage_actual = self.data.get_actual_prices(first_stage)
            scenario_profits = [np.dot(-np.array(prices), y)
                                for prices, y in zip(second_stage_scenarios, ys)]
            stats[first_stage] = {
                'cvar': cvar,
                'volume_traded': sum([abs(bid) for bid in bids]),
                'net_volume_traded': sum(bids),
                'profit': np.dot(-np.array(first_stage_actual), bids),
                'forecasted_profit': np.dot(-np.array(first_stage_forecast), bids),
                'SMAPE': self.data.calculate_smape(first_stage, first_stage_forecast),
                'scenario_stats': {
                    f'{second_stage}_profit_std': np.std(scenario_profits),
                    f'{second_stage}_max_profit': max(scenario_profits),
                    f'{second_stage}_min_profit': min(scenario_profits)
                }
            }

            self.data.realize_prices(first_stage)
            first_stage = second_stage
            second_stage = next_stages.pop(0) if next_stages else None
            print("\n")

        print("First stage:", first_stage)
        start_time, end_time = self.__get_trading_window(first_stage)
        first_stage_forecast = self.__forecast(first_stage)
        bids, final_soc = self.__solve_single(
            first_stage_forecast, prev_bids, start_time, end_time)
        schedules[first_stage] = bids

        self.data.realize_prices(first_stage)
        total_profit = self.__calculate_profit(schedules)

        stats[first_stage] = {
            'volume_traded': sum([abs(bid) for bid in bids[start_time:end_time]]),
            'net_volume_traded': sum(bids[start_time:end_time]),
            'profit': np.dot(-np.array(self.data.get_actual_prices(first_stage)), bids[start_time:end_time]),
            'forecasted_profit': np.dot(-np.array(first_stage_forecast), bids[start_time:end_time]),
            'SMAPE': self.data.calculate_smape(first_stage, first_stage_forecast)
        }

        cycles = np.abs(np.diff(final_soc)).sum() / (CAPACITY * 2)
        if SHOW_FINAL_SCHEDULE:
            realized_prices = [
                self.data.prediction_day[f'y_{auction}'] for auction in schedules.keys()]
            self.__show_schedule(schedules, realized_prices, final_soc)

        return stats, total_profit, cycles

    def __run_robust(self):
        print("Forecasting day:", self.data.date)

        first_stage = "DA"
        second_stage = "IDA1"
        next_stages = ["IDA2"]
        schedules = {}
        prev_bids = [0] * HORIZON
        stats = {}
        while second_stage:
            print("First stage:", first_stage)
            print("Second stage:", second_stage)

            first_stage_forecast = self.__forecast(first_stage)
            second_stage_scenarios = self.get_scenarios(
                self.__forecast(second_stage), second_stage, SCENARIO_TYPE)
            start_time_y, end_time_y = self.__get_trading_window(second_stage)

            print("BEST SCENARIO SMAPE:", min([
                self.data.calculate_smape(second_stage, second_stage_scenario)
                for second_stage_scenario in second_stage_scenarios
            ]))

            bids, ys, _, cvar, z = self.__solve_two_stage_robust(
                first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, prev_bids)
            schedules[first_stage] = bids
            prev_bids = np.add(prev_bids, bids)

            first_stage_actual = self.data.get_actual_prices(first_stage)
            scenario_profits = [np.dot(-np.array(prices), y)
                                for prices, y in zip(second_stage_scenarios, ys)]
            stats[first_stage] = {
                'cvar': cvar,
                'volume_traded': sum([abs(bid) for bid in bids]),
                'net_volume_traded': sum(bids),
                'profit': np.dot(-np.array(first_stage_actual), bids),
                'forecasted_profit': np.dot(-np.array(first_stage_forecast), bids),
                'SMAPE': self.data.calculate_smape(first_stage, first_stage_forecast),
                'scenario_stats': {
                    f'{second_stage}_profit_std': np.std(scenario_profits),
                    f'{second_stage}_max_profit': max(scenario_profits),
                    f'{second_stage}_min_profit': z
                }
            }

            self.data.realize_prices(first_stage)
            first_stage = second_stage
            second_stage = next_stages.pop(0) if next_stages else None
            print("\n")

        print("First stage:", first_stage)
        start_time, end_time = self.__get_trading_window(first_stage)
        first_stage_forecast = self.__forecast(first_stage)
        bids, final_soc = self.__solve_single(
            first_stage_forecast, prev_bids, start_time, end_time)
        schedules[first_stage] = bids

        self.data.realize_prices(first_stage)
        total_profit = self.__calculate_profit(schedules)

        stats[first_stage] = {
            'volume_traded': sum([abs(bid) for bid in bids[start_time:end_time]]),
            'net_volume_traded': sum(bids[start_time:end_time]),
            'profit': np.dot(-np.array(self.data.get_actual_prices(first_stage)), bids[start_time:end_time]),
            'forecasted_profit': np.dot(-np.array(first_stage_forecast), bids[start_time:end_time]),
            'SMAPE': self.data.calculate_smape(first_stage, first_stage_forecast)
        }

        if SHOW_FINAL_SCHEDULE:
            realized_prices = [
                self.data.prediction_day[f'y_{auction}'] for auction in schedules.keys()]
            self.__show_schedule(schedules, realized_prices, final_soc)

        cycles = np.abs(np.diff(final_soc)).sum() / (CAPACITY * 2)
        return stats, total_profit, cycles

    def __run_deterministic(self):
        first_stage = "DA"
        next_stages = ["IDA1", "IDA2"]
        schedule = {}
        stats = {}
        prev_bids = [0] * HORIZON
        while first_stage:

            forecast = self.__forecast(first_stage)
            start_time, end_time = self.__get_trading_window(first_stage)

            bids, soc = self.__solve_single(
                forecast, prev_bids, start_time, end_time)
            prev_bids = np.add(
                prev_bids, bids)
            schedule[first_stage] = bids

            self.data.realize_prices(first_stage)
            print(
                f"Actual Profit {first_stage}: {np.dot(-self.data.prediction_day[f'y_{first_stage}'].dropna(),bids[start_time:end_time])}")

            actual = self.data.get_actual_prices(first_stage)
            stats[first_stage] = {
                'bids': bids,
                'volume_traded': sum([abs(bid) for bid in bids]),
                'net_volume_traded': sum(bids),
                'profit': np.dot(-np.array(actual), bids[start_time:end_time]),
                'forecasted_profit': np.dot(-np.array(forecast), bids[start_time:end_time]),
                'SMAPE': self.data.calculate_smape(first_stage, forecast),
            }
            first_stage = next_stages.pop(0) if next_stages else None

        if SHOW_FINAL_SCHEDULE:
            realized_prices = [
                self.data.prediction_day[f'y_{auction}'] for auction in schedule.keys()]
            self.__show_schedule(schedule, realized_prices, soc)

        total_profit = self.__calculate_profit(schedule)
        cycles = np.abs(np.diff(soc)).sum() / (CAPACITY * 2)
        return stats, total_profit, cycles

    def run(self, num_days):
        results = {}
        for _ in range(num_days):
            print(f'\nRunning day {self.data.date}')
            print("=====================================\n")
            stages, profit, cycles = None, None, None
            if MODEL_TYPE == 'stochastic':
                stages, profit, cycles = self.__run_stochastic()
            elif MODEL_TYPE == 'deterministic':
                stages, profit, cycles = self.__run_deterministic()
            elif MODEL_TYPE == 'robust':
                stages, profit, cycles = self.__run_robust()
            else:
                raise ValueError("Invalid model type")

            results[self.data.date.strftime('%Y-%m-%d')] = {
                'profit': profit,
                'auctions': stages,
                'cycles': cycles
            }
            self.data.move_to_next_day()
            print("\n")

        return {"results": results}


if __name__ == '__main__':
    main()
