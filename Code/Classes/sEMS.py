from Classes.ems import EMS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from config import *


class StochasticEMS(EMS):
    def __init__(self, battery, auction_data):
        super().__init__(battery, auction_data)

    def generate_scenarios(self, forecast, auction):
        cross_val = self.auction_data.cross_vals[auction].copy()
        # cross_val = cross_val[cross_val['ds'].dt.date >= (
        #     self.auction_data.current_date - pd.Timedelta(days=30))]
        residuals = cross_val.groupby(
            [cross_val['ds'].dt.hour, cross_val['ds'].dt.minute])['error']

        quantiles = np.linspace(0, 1, NUM_SCENARIOS)
        scenarios = [forecast]
        for quantile in quantiles:
            errors = residuals.quantile(
                quantile).values
            scenario = np.add(forecast, errors)
            scenarios.append(scenario)

        if SHOW_SCENARIOS:
            self.__plot_scenarios(scenarios, auction)

        return scenarios

    def __plot_scenarios(self, scenarios, auction):
        date_mask = self.auction_data.testing_data[auction]['ds'].dt.date == self.auction_data.current_date
        real_prices = self.auction_data.testing_data[auction].loc[date_mask, [
            'y', 'ds']].dropna()
        plt.plot(real_prices['ds'], real_prices['y'], label='Real Prices')
        for scenario in scenarios:
            plt.plot(real_prices['ds'], scenario)
        plt.legend()
        plt.show()

    def __prepare_two_stage(self, first_stage, second_stage):
        first_stage_forecast = super().forecast(first_stage)
        second_stage_scenarios = self.generate_scenarios(
            super().forecast(second_stage), second_stage)

        start_time_y, end_time_y = self.auction_data.get_trading_window(
            second_stage)

        return first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y

    def __solve_two_stage(self, x_forecast, y_scenarios,
                          start_time_y, end_time_y, prev_bids,
                          risk_averse_factor=RISK_AVERSE_FACTOR):

        model = gp.Model("Master Combined")
        model.setParam('OutputFlag', 0)

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
        model.setObjective(((1-risk_averse_factor)*expected_profit) - (cvar *
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

    def __solve_final_stage(self, forecast, prev_bids, start_time_x, end_time_x):
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
            return (
                [x[t].x for t in range(HORIZON)],
                [soc[t+1].x for t in range(HORIZON)]
            )
        else:
            raise Exception("Master is infeasible")

    def run(self):
        first_stage, second_stage, next_stages = "DA", "IDA1", ["IDA2"]
        prev_bids = [0] * HORIZON
        stats = {}
        schedules = {}
        while second_stage:
            print(f"\nStages: \n1. {first_stage} \n2. {second_stage}")

            first_stage_forecast, second_stage_scenarios, start_time_y, end_time_y = self.__prepare_two_stage(
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
        start_time, end_time = self.auction_data.get_trading_window(
            first_stage)

        first_stage_forecast = super().forecast(first_stage)

        bids, final_soc = self.__solve_final_stage(
            first_stage_forecast, prev_bids, start_time, end_time)
        schedules[first_stage] = bids

        stats[first_stage] = super().get_stats(
            first_stage, first_stage_forecast, bids[start_time:end_time])

        self.auction_data.realize_prices(first_stage)
        total_profit = super().calculate_profit(schedules)
        cycles = np.abs(np.diff(final_soc)).sum() / (CAPACITY * 2)

        if SHOW_FINAL_SCHEDULE:
            super().plot_schedule(schedules, final_soc)

        return stats, total_profit, cycles
