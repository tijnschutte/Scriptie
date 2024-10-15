import json
import numpy as np
import pandas as pd
from statsforecast import StatsForecast
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

HORIZON = 48

POWER = 10  # MW
CAPACITY = 20  # MWh
MAX_TRADE = POWER / 2  # MW per half-hour
EFFICIENCY = 1

RISK_AVERSE_FACTOR = 0
BETA = 0.95


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

    def get_forecast(self, auction_name):
        exo_vars = self.training_data[auction_name].drop(
            columns=['y', 'ds', 'unique_id']).columns.tolist()

        print(
            f"Retrieving: ./Data/Forecasts/{auction_name}/{self.date}_exo{exo_vars}.xlsx")

        forecast = pd.read_excel(
            f'./Data/Forecasts/{auction_name}/{self.date}_exo{exo_vars}.xlsx',)
        return forecast['MSTL'].values

    def move_to_next_day(self):
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

    def add_prices_to_train(self, auction_name):
        date_mask = self.testing_data[auction_name]['ds'].dt.date == self.date
        auction_prices = self.testing_data[auction_name].loc[date_mask, 'y'].values
        self.prediction_day[f'y_{auction_name}'] = auction_prices

        auctions = ['DA', 'IDA1', 'IDA2']
        keys = auctions[auctions.index(auction_name) + 1:]
        for key in keys:
            print(f"Adding {auction_name} prices to {key} training data")
            self.training_data[key] = self.training_data[key].copy()
            merged_data = pd.merge(self.training_data[key],
                                   self.training_data[auction_name][[
                                       'ds', 'y']],
                                   on='ds',
                                   how='left',  # Keep all original rows for key
                                   suffixes=('', f'_{auction_name}'))
            self.training_data[key] = merged_data

    def get_real_prices(self, auction_name):
        date_mask = self.testing_data[auction_name]['ds'].dt.date == self.date
        return self.testing_data[auction_name].loc[date_mask, 'y'].dropna().values

    # def plot_scenarios(self, scenarios, auction_name):
    #     date_mask = self.testing_data[auction_name]['ds'].dt.date == self.date
    #     real_prices = self.testing_data[auction_name].loc[date_mask, ['y', 'ds']]
    #     plt.plot(real_prices['ds'], real_prices['y'], label='Real Prices')


class EMS:
    def __init__(self, data):
        self.data = data
        self.first_stage_forecast = []
        self.second_stage_scenarios = []

    def __forecast(self, auction_name):
        try:
            forecast = self.data.get_forecast(auction_name)
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

            print("train columns:", train.columns)
            print("X_df columns:", X_df.columns)
            print("horizon:", len(X_df))
            print("timeframe:", X_df['ds'].iloc[0], "to", X_df['ds'].iloc[-1])
            model = self.__create_mstl(
                seasons=[trading_length, 2*trading_length, 5*trading_length, 6*trading_length])
            forecast = model.forecast(
                df=train, h=len(X_df), X_df=X_df)

            forecast.reset_index(drop=True, inplace=True)
            forecast['ds'] = X_df['ds'].reset_index(drop=True)
            forecast.to_excel(
                f'./Data/Forecasts/{auction_name}/{self.data.date}_exo{exos}.xlsx', index=False)
            return forecast['MSTL'].values

    def __create_mstl(self, seasons=[48, 2*48, 5*48, 6*48]):
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

    def __generate_scenarios(self, predictions, auction_name):
        cross_val = self.data.cross_vals[auction_name]
        grouped = cross_val.groupby(
            [cross_val['ds'].dt.hour, cross_val['ds'].dt.minute])
        error_distr = grouped['error'].describe()

        errors_high = error_distr.apply(lambda row: np.percentile(np.random.normal(
            row['mean'], row['std'], 10000), 90), axis=1)
        errors_middle = error_distr.apply(lambda row: np.percentile(np.random.normal(
            row['mean'], row['std'], 10000), 50), axis=1)
        errors_low = error_distr.apply(lambda row: np.percentile(np.random.normal(
            row['mean'], row['std'], 10000), 10), axis=1)

        scenarios = np.vstack(
            [errors_high, errors_middle, errors_low]) + predictions

        shifted_scenarios = []
        for i in range(len(scenarios)):
            shifted_scenarios.append(np.roll(scenarios[i], shift=-2))

        scenario_set = np.vstack([scenarios, *shifted_scenarios])
        scenarios_df = pd.DataFrame(scenario_set).T

        return [scenarios_df[col].tolist() for col in scenarios_df.columns]

    def __get_start_times(self, auction_name):
        start_time_y = self.data.training_data[auction_name].dropna()[
            'ds'].dt.hour.min() * 2
        end_time_y = (self.data.training_data[auction_name].dropna()[
            'ds'].dt.hour.max() + 1) * 2
        return start_time_y, end_time_y

    def __calc_profit(self, schedules):
        profit = 0
        for auction in ['DA', 'IDA1', 'IDA2']:
            if auction in schedules:
                date_mask = self.data.testing_data[auction]['ds'].dt.date == self.data.date
                actual_prices = self.data.testing_data[auction].loc[date_mask, 'y'].dropna(
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

    def __plot_cvars(self, first_stage, second_stage, soc):
        res = {}
        first_stage_forecast = self.__forecast(first_stage)
        second_stage_scenarios = self.__generate_scenarios(
            self.__forecast(second_stage), second_stage)
        start_time_y, end_time_y = self.__get_start_times(second_stage)

        for l in np.linspace(0, 1, 11):
            print(f"Risk Averse Factor: {l}")
            bids, adjustments, _, cvar, e_scenarios_profit = self.__solve_cvar(
                first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, soc, risk_averse_factor=l)

            scenario_profits = [np.dot(-np.array(first_stage_forecast), bids) + np.dot(-np.array(prices), trades)
                                for trades, prices in zip(adjustments, second_stage_scenarios)]

            res[l] = {
                "cvar": cvar,
                "worst-case": min(scenario_profits),
                "best-case": max(scenario_profits),
                "std": np.std(scenario_profits),
                "expected profit": np.dot(-np.array(first_stage_forecast), bids) +
                e_scenarios_profit,
            }
            for key, value in res[l].items():
                print(f"{key}: {value}")
            print('\n')

        risk_averse_factors = list(res.keys())
        worst_cases = [res[l]["worst-case"] for l in risk_averse_factors]
        cvars = [res[l]["cvar"] for l in risk_averse_factors]
        expected_profits = [res[l]["expected profit"]
                            for l in risk_averse_factors]
        plt.plot(risk_averse_factors, cvars, label="CVaR")
        plt.plot(risk_averse_factors, worst_cases, label="Worst-case")
        plt.plot(risk_averse_factors, expected_profits,
                 label="Expected Profit")
        plt.title(f"Battery: {POWER} MW, {CAPACITY} MWh")
        plt.xlabel("Risk Averse Factor")
        plt.ylabel("Value (eur)")
        plt.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        plt.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        plt.legend()
        plt.show()

    def __solve_cvar(self, x_forecasted, y_scenarios,
                     start_time_y, end_time_y, prev_bids=[0]*HORIZON,
                     risk_averse_factor=RISK_AVERSE_FACTOR):

        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)  # Turn off Gurobi output
        N = np.array(y_scenarios).shape[0]
        # print("trading length y:", np.array(y_scenarios).shape[1])
        # print("start time y:", start_time_y)
        # print("end time y:", end_time_y)

        x = model.addVars(HORIZON, lb=-MAX_TRADE,
                          ub=MAX_TRADE, name="da_schedule")
        y = model.addVars(HORIZON, N, lb=-MAX_TRADE, ub=MAX_TRADE,
                          name="ida_adjustments")
        soc = model.addVars(HORIZON+1, N, lb=0, ub=CAPACITY, name="soc")

        for s in range(N):
            model.addConstr(soc[0, s] == 0, name=f"initial_soc_s{s}")

        for t in range(HORIZON):
            for s in range(N):
                if t < start_time_y or t > end_time_y:
                    model.addConstr(y[t, s] == 0)

                model.addConstr(
                    soc[t+1, s] == soc[t, s] +
                    (EFFICIENCY * x[t]) +
                    (EFFICIENCY * y[t, s]) + prev_bids[t],
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
            -y_scenarios[s][t - start_time_y] * y[t, s] for t in range(start_time_y, end_time_y)) for s in range(N)}

        alpha = model.addVar(lb=0, name="VaR")
        z = model.addVars(N, lb=0, name="excess_loss")
        loss = model.addVars(N, lb=0, name="loss")

        expected_scenario_profit = (
            1/N)*gp.quicksum(scenario_profit[s] for s in range(N))

        for s in range(N):
            model.addConstr(
                loss[s] >= expected_scenario_profit - scenario_profit[s])
            model.addConstr(z[s] >= loss[s] - alpha)

        cvar = model.addVar(lb=0, name="cvar")

        model.addConstr(cvar >= alpha + (1 / (N * (1 - BETA)))
                        * gp.quicksum(z[s] for s in range(N)))

        expected_profit = gp.quicksum(-x_forecasted[t] * x[t]
                                      for t in range(HORIZON)) + (1/N)*gp.quicksum(scenario_profit[s] for s in range(N))
        model.setObjective(expected_profit - cvar *
                           risk_averse_factor, GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return (
                [x[t].x for t in range(HORIZON)],
                [[y[t, s].x for t in range(start_time_y, end_time_y)]
                 for s in range(N)],
                [[soc[t+1, s].x for t in range(HORIZON)] for s in range(N)],
                cvar.X,
                expected_scenario_profit.getValue()
            )
        else:
            raise Exception("Master is infeasible")

    def __solve_single(self, x_forecasted, prev_bids, start_time_x, end_time_x):
        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)  # Turn off Gurobi output

        print("Trading period length:", len(x_forecasted))
        print("start time x:", start_time_x)
        print("end time x:", end_time_x)

        x = model.addVars(HORIZON, lb=-MAX_TRADE,
                          ub=MAX_TRADE, name="da_schedule")
        soc = model.addVars(HORIZON+1, lb=0, ub=CAPACITY, name="soc")

        model.addConstr(soc[0] == 0, name=f"initial_soc")

        for t in range(HORIZON):
            if t < start_time_x or t > end_time_x:
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

        profit = gp.quicksum(-(x_forecasted[t - start_time_x] * x[t])
                             for t in range(start_time_x, end_time_x))

        model.setObjective(profit, gp.GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return (
                [x[t].x for t in range(HORIZON)],
                [soc[t+1].x for t in range(HORIZON)]
            )
        else:
            raise Exception("Master is infeasible")

    def run_day(self):
        print("Forecasting day:", self.data.date)

        first_stage = "DA"
        second_stage = "IDA1"
        next_stages = ["IDA2"]
        schedules = {}
        soc = [0] * HORIZON
        stats = {}
        while second_stage:
            print("First stage:", first_stage)
            print("Second stage:", second_stage)

            first_stage_forecast = self.__forecast(first_stage)
            second_stage_scenarios = self.__generate_scenarios(
                self.__forecast(second_stage), second_stage)
            first_stage_actual = self.data.get_real_prices(first_stage)
            second_stage_actual = self.data.get_real_prices(second_stage)

            start_time_y, end_time_y = self.__get_start_times(second_stage)

            # self.__plot_cvars(first_stage, second_stage, soc)
            bids, ys, soc_ys, cvar, _ = self.__solve_cvar(
                first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, soc, RISK_AVERSE_FACTOR)
            soc = np.add(soc, bids)

            # for y, prices, soc in zip(ys, second_stage_scenarios, soc_ys):
            #     bids_y = [0] * start_time_y + y + [0] * (HORIZON - end_time_y)
            #     price_scenario = [np.nan] * start_time_y + \
            #         prices + [np.nan]*(HORIZON - end_time_y)
            #     self.__show_schedule({first_stage: bids, second_stage: bids_y},
            #                          [first_stage_forecast, price_scenario],
            #                          soc)

            schedules[first_stage] = bids
            self.data.add_prices_to_train(first_stage)
            first_stage = second_stage
            second_stage = next_stages.pop(0) if next_stages else None

            scenario_profits = [np.dot(-np.array(prices), y)
                                for prices, y in zip(second_stage_scenarios, ys)]
            stats[first_stage] = {
                'cvar': cvar,
                'actual_profit': np.dot(-np.array(first_stage_actual), bids),
                'forecasted_profit': np.dot(-np.array(first_stage_forecast), bids),
                'scenario_stats': {
                    'total_profit_std': np.std(scenario_profits),
                    'total_max_profit': max(scenario_profits),
                    'total_min_profit': min(scenario_profits)
                }
            }
            print("\n")

        print("First stage:", first_stage)
        start_time, end_time = self.__get_start_times(first_stage)
        bids, final_soc = self.__solve_single(
            self.__forecast(first_stage), soc, start_time, end_time)
        schedules[first_stage] = bids[start_time:end_time]
        self.data.add_prices_to_train(first_stage)

        total_profit = self.__calc_profit(schedules)
        schedules[first_stage] = bids

        realized_prices = [
            self.data.prediction_day[f'y_{auction}'] for auction in schedules.keys()]
        # self.__show_schedule(schedules, realized_prices, final_soc)

        print("\n")
        return stats, total_profit

    def run(self, num_days):
        results = {}
        for i in range(num_days):
            print(f'\nRunning day {self.data.date}')
            print("=====================================\n")
            stages, profit = self.run_day()
            results[self.data.date.strftime('%Y-%m-%d')] = {
                'overall_profit': profit,
                'stages': stages
            }
            self.data.move_to_next_day()
        return results


ems = EMS(Data())
results = ems.run(10)
pretty_printed = json.dumps(results, indent=4)
print(pretty_printed)
