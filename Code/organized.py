import json
import os
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
pd.options.display.float_format = '{:.2f}'.format
# from abc import ABC, abstractmethod

HORIZON = 48

POWER = 10  # MW
CAPACITY = 20  # MWh
MAX_TRADE = POWER / 2  # MW per half-hour
EFFICIENCY = 1

# 'stochastic', 'deterministic'
MODEL_TYPE = 'stochastic'
RISK_AVERSE_FACTOR = 0
BETA = 0.95
NUM_SCENARIOS = 100

SHOW_FINAL_SCHEDULE = False
CVAR_PLOT_DAY = None
SHOW_SCENARIOS = False



# def main():
#     ems = EMS(Data())
#     max_days = ems.data.testing_data['DA'].shape[0] // 48

#     results = ems.run(max_days)
#     results['params'] = {
#         'POWER': POWER,
#         'CAPACITY': CAPACITY,
#         'MAX_TRADE': MAX_TRADE,
#         'EFFICIENCY': EFFICIENCY,
#         'RISK_AVERSE_FACTOR': RISK_AVERSE_FACTOR,
#         'BETA': BETA,
#         'NUM_SCENARIOS': NUM_SCENARIOS,
#     }

#     results_dir = 'results'
#     os.makedirs(results_dir, exist_ok=True)
#     subfolder_name = f'{MODEL_TYPE}/battery_{POWER}MW_{CAPACITY}MWh'
#     subfolder_path = os.path.join(results_dir, subfolder_name)
#     if not os.path.exists(subfolder_path):
#         os.makedirs(subfolder_path)
#     subfolder_path = os.path.join(results_dir, subfolder_name)
#     os.makedirs(subfolder_path, exist_ok=True)
#     if MODEL_TYPE == 'stochastic':
#         results_file = os.path.join(
#             subfolder_path, f'run_results_l={RISK_AVERSE_FACTOR}.json')
#     else:
#         results_file = os.path.join(subfolder_path, 'run_results.json')
#     with open(results_file, 'w') as f:
#         json.dump(results, f, indent=4)


# class Data:

#     def insert_spike(self, auction):
#         date_mask = self.testing_data[auction]['ds'].dt.date == self.date
#         self.testing_data[auction].loc[date_mask, 'y']
#         max_index = self.testing_data[auction].loc[date_mask, 'y'].idxmax(
#         )
#         self.testing_data[auction].loc[max_index, 'y'] *= 0.25
#         print("Spike inserted at",
#               self.testing_data[auction].loc[max_index, 'ds'])

#     def get_forecast_from_file(self, auction):
#         exo_vars = self.training_data[auction].drop(
#             columns=['y', 'ds', 'unique_id']).columns.tolist()

#         print(
#             f"Retrieving: ./Data/Forecasts/{auction}/{self.date}_exo{exo_vars}.xlsx")

#         forecast = pd.read_excel(
#             f'./Data/Forecasts/{auction}/{self.date}_exo{exo_vars}.xlsx')

#         return forecast['MSTL'].values

# class EMS:
#     def __init__(self, data):
#         self.data = data

#     def __forecast(self, auction):
#         try:
#             forecast = self.data.get_forecast_from_file(auction)
#             print("SMAPE:", self.data.calculate_smape(auction, forecast))
#             return forecast
#         except:
#             train = self.data.training_data[auction]
#             train.dropna(inplace=True)
#             train.reset_index(drop=True, inplace=True)
#             exos = train.drop(
#                 columns=['y', 'ds', 'unique_id']).columns.tolist()
#             print(
#                 f"FILE NOT FOUND: Forecasting for {auction} on {self.data.date} using exogeneous variables: {exos}")
#             trading_hours = train['ds'].dt.hour.unique()

#             X_df = self.data.prediction_day[train.drop(columns=['y']).columns]
#             timeframe_mask = X_df['ds'].dt.floor(
#                 'h').dt.hour.isin(trading_hours)
#             X_df = X_df.loc[timeframe_mask]

#             trading_length = len(trading_hours) * 2

#             print("timeframe:", X_df['ds'].iloc[0], "to", X_df['ds'].iloc[-1])
#             model = self.__create_mstl(
#                 seasons=[trading_length, 2*trading_length])
#             forecast = model.forecast(
#                 df=train, h=len(X_df), X_df=X_df)

#             forecast.reset_index(drop=True, inplace=True)
#             forecast['ds'] = X_df['ds'].reset_index(drop=True)

#             print("SMAPE:", self.data.calculate_smape(
#                 auction, forecast['MSTL'].values))
#             forecast.to_excel(
#                 f'./Data/Forecasts/{auction}/{self.data.date}_exo{exos}.xlsx', index=False)
#             return forecast['MSTL'].values

#     def __plot_cvars(self, first_stage_forecast, second_stage_scenarios,
#                      start_time_y, end_time_y, soc):
#         print("Day:", self.data.date)
#         res = {}
#         # for l in np.linspace(0, 1, 11):
#         for l in [0, 0.25, 0.5, 0.75, 1]:
#             bids, adjustments, _, cvar, expected_profit = self.__solve_two_stage(
#                 first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, soc, risk_averse_factor=l)

#             scenario_profits = [np.dot(-np.array(first_stage_forecast), bids) + np.dot(-np.array(
#                 prices), trades) for trades, prices in zip(adjustments, second_stage_scenarios)]

#             res[l] = {
#                 "cvar": cvar,
#                 "expected": expected_profit,
#                 "worst": min(scenario_profits),
#                 "std": np.std(scenario_profits),
#             }
#         risk_averse_factors = list(res.keys())
#         cvars = [res[l]["cvar"] for l in risk_averse_factors]
#         print("cvars = ", cvars)
#         expected = [res[l]["expected"] for l in risk_averse_factors]
#         print("expected = ", expected)
#         worst = [res[l]["worst"] for l in risk_averse_factors]
#         print("worst = ", worst)
#         std = [res[l]["std"] for l in risk_averse_factors]
#         plt.plot(risk_averse_factors, cvars, label="CVaR")
#         plt.plot(risk_averse_factors, expected, label="Expected Profit")
#         # plt.plot(risk_averse_factors, std, label="Scenario Profit Std")
#         plt.plot(risk_averse_factors, worst, label="Worst-Case Profit")
#         plt.title(f"Battery: {POWER} MW, {CAPACITY} MWh")
#         plt.xlabel("Risk Averse Factor")
#         plt.ylabel("Value (eur)")
#         plt.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
#         plt.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
#         plt.legend()
#         plt.show()

#

#     def __run_stochastic(self):
#         print("Forecasting day:", self.data.date)

#         first_stage = "DA"
#         second_stage = "IDA1"
#         next_stages = ["IDA2"]
#         schedules = {}
#         prev_bids = [0] * HORIZON
#         stats = {}
#         while second_stage:
#             print("First stage:", first_stage)
#             print("Second stage:", second_stage)

#             first_stage_forecast = self.__forecast(first_stage)
#             second_stage_scenarios = self.__get_dexter_scenarios(
#                 self.__forecast(second_stage), second_stage)

#             start_time_y, end_time_y = self.__get_trading_window(second_stage)

#             print("BEST SCENARIO SMAPE:", min([
#                 self.data.calculate_smape(second_stage, second_stage_scenario)
#                 for second_stage_scenario in second_stage_scenarios
#             ]))

#             if first_stage == "DA" and self.data.date.day == CVAR_PLOT_DAY:
#                 self.__plot_cvars(first_stage_forecast, second_stage_scenarios, start_time_y,
#                                   end_time_y, prev_bids)

#             bids, ys, _, cvar, _ = self.__solve_two_stage(
#                 first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, prev_bids)
#             schedules[first_stage] = bids
#             prev_bids = np.add(prev_bids, bids)

#             first_stage_actual = self.data.get_actual_prices(first_stage)
#             scenario_profits = [np.dot(-np.array(prices), y)
#                                 for prices, y in zip(second_stage_scenarios, ys)]
#             stats[first_stage] = {
#                 'cvar': cvar,
#                 'volume_traded': sum([abs(bid) for bid in bids]),
#                 'net_volume_traded': sum(bids),
#                 'profit': np.dot(-np.array(first_stage_actual), bids),
#                 'forecasted_profit': np.dot(-np.array(first_stage_forecast), bids),
#                 'SMAPE': self.data.calculate_smape(first_stage, first_stage_forecast),
#                 'scenario_stats': {
#                     f'{second_stage}_profit_std': np.std(scenario_profits),
#                     f'{second_stage}_max_profit': max(scenario_profits),
#                     f'{second_stage}_min_profit': min(scenario_profits)
#                 }
#             }

#             self.data.realize_prices(first_stage)
#             first_stage = second_stage
#             second_stage = next_stages.pop(0) if next_stages else None
#             print("\n")

#         print("First stage:", first_stage)
#         start_time, end_time = self.__get_trading_window(first_stage)
#         first_stage_forecast = self.__forecast(first_stage)
#         bids, final_soc = self.__solve_single(
#             first_stage_forecast, prev_bids, start_time, end_time)
#         schedules[first_stage] = bids

#         self.data.realize_prices(first_stage)
#         total_profit = self.__calculate_profit(schedules)

#         stats[first_stage] = {
#             'volume_traded': sum([abs(bid) for bid in bids[start_time:end_time]]),
#             'net_volume_traded': sum(bids[start_time:end_time]),
#             'profit': np.dot(-np.array(self.data.get_actual_prices(first_stage)), bids[start_time:end_time]),
#             'forecasted_profit': np.dot(-np.array(first_stage_forecast), bids[start_time:end_time]),
#             'SMAPE': self.data.calculate_smape(first_stage, first_stage_forecast)
#         }

#         cycles = np.abs(np.diff(final_soc)).sum() / (CAPACITY * 2)
#         if SHOW_FINAL_SCHEDULE:
#             realized_prices = [
#                 self.data.prediction_day[f'y_{auction}'] for auction in schedules.keys()]
#             self.__show_schedule(schedules, realized_prices, final_soc)

#         return stats, total_profit, cycles

#     def __run_deterministic(self):
#         first_stage = "DA"
#         next_stages = ["IDA1", "IDA2"]
#         schedule = {}
#         stats = {}
#         prev_bids = [0] * HORIZON
#         while first_stage:

#             forecast = self.__forecast(first_stage)
#             start_time, end_time = self.__get_trading_window(first_stage)

#             bids, soc = self.__solve_single(
#                 forecast, prev_bids, start_time, end_time)
#             prev_bids = np.add(
#                 prev_bids, bids)
#             schedule[first_stage] = bids

#             self.data.realize_prices(first_stage)
#             print(
#                 f"Actual Profit {first_stage}: {np.dot(-self.data.prediction_day[f'y_{first_stage}'].dropna(),bids[start_time:end_time])}")

#             actual = self.data.get_actual_prices(first_stage)
#             stats[first_stage] = {
#                 'bids': bids,
#                 'volume_traded': sum([abs(bid) for bid in bids]),
#                 'net_volume_traded': sum(bids),
#                 'profit': np.dot(-np.array(actual), bids[start_time:end_time]),
#                 'forecasted_profit': np.dot(-np.array(forecast), bids[start_time:end_time]),
#                 'SMAPE': self.data.calculate_smape(first_stage, forecast),
#             }
#             first_stage = next_stages.pop(0) if next_stages else None

#         if SHOW_FINAL_SCHEDULE:
#             realized_prices = [
#                 self.data.prediction_day[f'y_{auction}'] for auction in schedule.keys()]
#             self.__show_schedule(schedule, realized_prices, soc)

#         total_profit = self.__calculate_profit(schedule)
#         cycles = np.abs(np.diff(soc)).sum() / (CAPACITY * 2)
#         return stats, total_profit, cycles

#     def run(self, num_days):
#         results = {}
#         for _ in range(num_days):
#             print(f'\nRunning day {self.data.date}')
#             print("=====================================\n")
#             stages, profit, cycles = None, None, None
#             if MODEL_TYPE == 'stochastic':
#                 stages, profit, cycles = self.__run_stochastic()
#             elif MODEL_TYPE == 'deterministic':
#                 stages, profit, cycles = self.__run_deterministic()
#             else:
#                 raise ValueError("Invalid model type")

#             results[self.data.date.strftime('%Y-%m-%d')] = {
#                 'profit': profit,
#                 'auctions': stages,
#                 'cycles': cycles
#             }
#             self.data.move_to_next_day()
#             print("\n")

#         return {"results": results}


# if __name__ == '__main__':
#     main()
