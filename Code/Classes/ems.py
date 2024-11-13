import gurobipy as gp
from gurobipy import GRB
from statsforecast.models import MSTL, AutoARIMA
from statsforecast import StatsForecast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *


class EMS:
    def __init__(self, battery, auction_data, risk_averse_factor=None):
        self.auction_data = auction_data
        self.battery = battery
        self.risk_averse_factor = risk_averse_factor

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

    def forecast(self, auction):
        try:
            forecast = self.auction_data.get_forecast_from_file(auction)
            rating = self.rate_forecast(auction, forecast)
            print(f"MNAE: {rating:.2f}%")
            return forecast
        except:
            train = self.auction_data.training_data[auction]
            train.dropna(inplace=True)
            train.reset_index(drop=True, inplace=True)
            exos = train.drop(
                columns=['y', 'ds', 'unique_id']).columns.tolist()
            print(
                f"FILE NOT FOUND: Forecasting for {auction} on {self.auction_data.current_date} using exogeneous variables: {exos}")
            trading_hours = train['ds'].dt.hour.unique()

            X_df = self.auction_data.prediction_day[train.drop(
                columns=['y']).columns]
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

            forecast.to_excel(
                f'./Data/Forecasts/{auction}/{self.auction_data.current_date}_exo{exos}.xlsx', index=False)
            rating = self.rate_forecast(auction, forecast['MSTL'].values)
            print(f'{rating:.2f}%')
            return forecast['MSTL'].values

    def plot_schedule(self, schedules, soc_levels):
        # fixme:
        _, (price_ax, power_ax) = plt.subplots(2, 1, sharex=True)
        common_index = self.auction_data.prediction_day['ds']

        schedule = pd.DataFrame(index=common_index)
        for auction, bids in schedules.items():
            bid_series = pd.Series(bids, index=common_index)
            schedule[auction] = bid_series.reindex(
                common_index, fill_value=None)
        schedule.plot(kind='bar', stacked=True, ax=power_ax)

        power_ax.plot(soc_levels, label='SOC', color='orange', linewidth=2)

        price_df = pd.DataFrame(index=common_index)
        for auction in schedules.keys():
            auction_prices = self.auction_data.get_auction_prices(auction)
            start_time, end_time = self.auction_data.get_trading_window(
                auction)
            price_series = pd.Series(
                auction_prices, index=common_index[start_time:end_time])
            price_series = price_series.reindex(common_index, fill_value=None)
            price_df[auction] = price_series
        price_df.plot(ax=price_ax)

        plt.suptitle(
            f"Battery: {self.battery.power} MW, {self.battery.capacity} MWh")
        plt.xticks(range(0, 48, 8), range(0, 24, 4))
        plt.xlabel("Time (hours)")
        power_ax.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        power_ax.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        power_ax.legend()
        price_ax.set_ylabel("Price (eur)")
        price_ax.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        price_ax.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
        price_ax.legend()
        plt.show()

    def calculate_profit(self, schedules):
        profit = 0
        for auction, schedule in schedules.items():
            prices = self.auction_data.get_auction_prices(auction)
            print(len(prices), len(schedule))
            if len(prices) != len(schedule):
                print(prices)
                print(schedule)
                raise Exception(
                    f"Length of prices ({len(prices)}) and schedule ({len(schedule)}) do not match")

            profit += np.dot(-np.array(prices), schedule)
        return profit

    def get_stats(self, auction, forecast, bids, cvar=0):
        actual = self.auction_data.get_auction_prices(auction)
        return {
            'cvar': cvar,
            'volume_traded': sum([abs(bid) for bid in bids]),
            'net_volume_traded': sum(bids),
            'profit': np.dot(-np.array(actual), bids),
            'forecasted_profit': np.dot(-np.array(forecast), bids),
            'SMAPE': self.rate_forecast(auction, forecast),
            'bids': bids,
            'actual_prices': actual,
            'forecast': forecast
        }

    def rate_forecast(self, auction, forecast):
        date_mask = self.auction_data.testing_data[auction]['ds'].dt.date == self.auction_data.current_date
        actual = self.auction_data.testing_data[auction].loc[date_mask, 'y'].dropna(
        ).values
        nmae = np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual)) * 100
        return nmae

    def generate_scenarios(self, forecast, auction):
        cross_val = self.auction_data.cross_vals[auction].copy()
        # cross_val = cross_val[cross_val['ds'].dt.date >= (
        #     self.auction_data.current_date - pd.Timedelta(days=30))]
        residuals = cross_val.groupby(
            [cross_val['ds'].dt.hour, cross_val['ds'].dt.minute])['error']

        quantiles = np.linspace(0.05, 0.95, NUM_SCENARIOS)
        scenarios = [forecast]
        for quantile in quantiles:
            errors = residuals.quantile(
                quantile).values
            scenario = np.add(forecast, errors)
            scenarios.append(scenario)

        if SHOW_SCENARIOS:
            self.plot_scenarios(scenarios, auction)

        return scenarios

    def plot_scenarios(self, scenarios, auction):
        date_mask = self.auction_data.testing_data[auction]['ds'].dt.date == self.auction_data.current_date
        real_prices = self.auction_data.testing_data[auction].loc[date_mask, [
            'y', 'ds']].dropna()
        plt.plot(real_prices['ds'], real_prices['y'], label='Real Prices')
        for scenario in scenarios:
            plt.plot(real_prices['ds'], scenario)
        plt.legend()
        plt.show()

    def prepare_two_stage(self, first_stage, second_stage):
        first_stage_forecast = self.forecast(first_stage)
        second_stage_scenarios = self.generate_scenarios(
            self.forecast(second_stage), second_stage)

        start_time_y, end_time_y = self.auction_data.get_trading_window(
            second_stage)

        return first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y


class StochasticEMS(EMS):
    def __init__(self, battery, auction_data, risk_averse_factor):
        super().__init__(battery, auction_data, risk_averse_factor)

    def __solve_two_stage(self, x_forecast, y_scenarios,
                          start_time_y, end_time_y, prev_bids):

        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)

        x = model.addVars(HORIZON, lb=-self.battery.max_trade,
                          ub=self.battery.max_trade, name="bids")
        y = model.addVars(HORIZON, NUM_SCENARIOS, lb=-self.battery.max_trade, ub=self.battery.max_trade,
                          name="adjustments")
        soc = model.addVars(HORIZON+1, NUM_SCENARIOS,
                            lb=0, ub=self.battery.capacity, name="soc")

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
                    soc[t+1, s] <= soc[t, s] + self.battery.power,
                    name=f"physical_charge_limit_t{t}_s{s}"
                )
                model.addConstr(
                    soc[t+1, s] >= soc[t, s] - self.battery.power,
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
        model.addConstr(cvar >= alpha + ((1 / (NUM_SCENARIOS * (1 - BETA)))
                        * gp.quicksum(z[s] for s in range(NUM_SCENARIOS))))

        expected_profit = gp.quicksum(
            (-x_forecast[t] * x[t]) for t in range(HORIZON)) + expected_scenario_profit
        model.setObjective(((1-self.risk_averse_factor)*expected_profit) - (cvar *
                           self.risk_averse_factor), GRB.MAXIMIZE)
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
            raise Exception(f"Master failed with status {model.status}")

    def __solve_final_stage(self, forecast, prev_bids, start_time_x, end_time_x):
        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)  # Turn off Gurobi output

        x = model.addVars(HORIZON, lb=-self.battery.max_trade,
                          ub=self.battery.max_trade, name="bids")
        soc = model.addVars(
            HORIZON+1, lb=0, ub=self.battery.capacity, name="soc")

        model.addConstr(soc[0] == 0, name="initial_soc")

        for t in range(HORIZON):
            if t < start_time_x or t >= end_time_x:
                model.addConstr(x[t] == 0)

            model.addConstr(
                soc[t+1] == soc[t] + x[t] + prev_bids[t],
                name=f"soc_balance_t{t}"
            )
            model.addConstr(
                soc[t+1] <= soc[t] + self.battery.power,
                name=f"physical_charge_limit_t{t}"
            )
            model.addConstr(
                soc[t+1] >= soc[t] - self.battery.power,
                name=f"physical_discharge_limit_t{t}"
            )

        profit = gp.quicksum(-forecast[t - start_time_x] * x[t]
                             for t in range(start_time_x, end_time_x))

        model.setObjective(profit, GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return (
                [x[t].x for t in range(start_time_x, end_time_x)],
                [soc[t+1].x for t in range(HORIZON)]
            )
        else:
            raise Exception(f"Master failed with status {model.status}")

    def run(self):
        first_stage, second_stage, next_stages = "DA", "IDA1", ["IDA2"]
        prev_bids = [0] * HORIZON
        stats = {}
        schedules = {}
        while second_stage:
            print(f"\nStages: \n1. {first_stage} \n2. {second_stage}")

            first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y = super().prepare_two_stage(
                first_stage, second_stage)

            bids, _, _, cvar, _ = self.__solve_two_stage(
                first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y, prev_bids)
            schedules[first_stage] = bids
            prev_bids = np.add(prev_bids, bids)

            stats[first_stage] = super().get_stats(
                first_stage, first_stage_forecast, bids, cvar)

            self.auction_data.realize_prices(first_stage)
            first_stage = second_stage
            second_stage = next_stages.pop(0) if next_stages else None

        print(f"\nStages: \n1. {first_stage}")

        forecast = super().forecast(first_stage)
        start_time, end_time = self.auction_data.get_trading_window(
            first_stage)

        bids, final_soc = self.__solve_final_stage(
            forecast, prev_bids, start_time, end_time)
        schedules[first_stage] = bids
        if max(final_soc) > self.battery.capacity:
            raise Exception("Battery capacity ub exceeded")
        elif min(final_soc) < 0:
            raise Exception("Battery capacity lb exceeded")

        stats[first_stage] = super().get_stats(
            first_stage, forecast, bids)

        self.auction_data.realize_prices(first_stage)
        total_profit = super().calculate_profit(schedules)
        cycles = np.abs(np.diff(final_soc)).sum() / (self.battery.capacity * 2)

        if SHOW_FINAL_SCHEDULE:
            super().plot_schedule(schedules, final_soc)

        return stats, total_profit, cycles


class DeterministicEMS(EMS):
    def __init__(self, battery, auction_data):
        super().__init__(battery, auction_data)

    def __solve_stage(self, forecast, prev_bids, start_time, end_time):
        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)  # Turn off Gurobi output

        x = model.addVars(HORIZON, lb=-self.battery.max_trade,
                          ub=self.battery.max_trade, name="bids")
        soc = model.addVars(
            HORIZON+1, lb=0, ub=self.battery.capacity, name="soc")

        model.addConstr(soc[0] == 0, name="initial_soc")

        for t in range(HORIZON):
            if t < start_time or t >= end_time:
                model.addConstr(x[t] == 0)

            model.addConstr(
                soc[t+1] == soc[t] + x[t] + prev_bids[t],
                name=f"soc_balance_t{t}"
            )
            model.addConstr(
                soc[t+1] <= soc[t] + self.battery.power,
                name=f"physical_charge_limit_t{t}"
            )
            model.addConstr(
                soc[t+1] >= soc[t] - self.battery.power,
                name=f"physical_discharge_limit_t{t}"
            )

        profit = gp.quicksum(-forecast[t - start_time] * x[t]
                             for t in range(start_time, end_time))

        model.setObjective(profit, GRB.MAXIMIZE)
        model.optimize()

        if model.status == GRB.OPTIMAL:
            return [x[t].x for t in range(start_time, end_time)],
        else:
            raise Exception(f"Master failed with status {model.status}")

    def run(self):
        prev_bids = [0] * HORIZON
        stats = {}
        schedules = {}
        for stage in ["DA", "IDA1", "IDA2"]:
            print(f"\nCurrent auction: {stage}")

            forecast = super().forecast(stage)
            start_time, end_time = self.auction_data.get_trading_window(stage)

            bids = self.__solve_stage(
                forecast, prev_bids, start_time, end_time)

            schedules[stage] = bids
            prev_bids[start_time:end_time] = np.add(
                prev_bids[start_time:end_time], bids)

            stats[stage] = super().get_stats(
                stage, forecast, bids)

            self.auction_data.realize_prices(stage)

        final_soc = np.cumsum(prev_bids)
        if max(final_soc) > self.battery.capacity:
            raise Exception("Battery capacity ub exceeded")
        elif min(final_soc) < 0:
            raise Exception("Battery capacity lb exceeded")

        total_profit = super().calculate_profit(schedules)
        cycles = np.abs(np.diff(final_soc)).sum() / (self.battery.capacity * 2)

        if SHOW_FINAL_SCHEDULE:
            super().plot_schedule(schedules, final_soc)

        return stats, total_profit, cycles
