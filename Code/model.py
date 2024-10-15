import pandas as pd
import numpy as np
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
DA_PRICES_PATH = 'Data/DA_forecast.xlsx'
IDA1_SCENARIOS_PATH = 'Data/scenarios.xlsx'
IDA1_SCENARIOS_WITH_WEIGHTS_PATH = 'Data/scenarios_with_weights.xlsx'

TIME = 48

POWER = 10  # MW
CAPACITY = 20  # MWh
MAX_TRADE = POWER / 2  # MW per half-hour
EFFICIENCY = 1

RISK_AVERSE_FACTOR = 1
BETA = 0.95

FIND_BEST_LAMBDA = True
PLOT_SCENARIO_RESULTS = False
PLOT_FINAL_RESULTS = False


def show_schedule(x, y, prices_x, prices_y, soc_levels):
    _, (price_ax, power_ax) = plt.subplots(2, 1)

    schedule = pd.DataFrame([x, y]).T
    schedule.columns = ['A1', 'A2']
    schedule.plot(kind='bar', stacked=True, ax=power_ax)
    power_ax.plot(soc_levels, color='yellow', label='SOC')

    price_ax.plot(prices_x, label='A1 Prices')
    price_ax.plot(prices_y, label='A2 Prices')
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


def get_max_adjustments_y(std_errors):
    normalized_confidence = [se / max(std_errors) for se in std_errors]
    adjustment_factors = [np.e**(-se) for se in normalized_confidence]
    return [MAX_TRADE * factor for factor in adjustment_factors]


def plot_scenario_results(a1_schedule, adjustments, da_actual, ida_scenarios, soc_scenarios):
    da_profit = np.dot(-da_actual, a1_schedule)
    print("Actual profit in Auction 1:", da_profit, "\n")
    for i, (a2_trades, prices, soc) in enumerate(zip(adjustments, ida_scenarios, soc_scenarios)):
        print(f"Total profit (scenario {i}):",
              da_profit + np.dot(-prices, a2_trades))
        if PLOT_SCENARIO_RESULTS:
            show_schedule(a1_schedule, a2_trades, da_actual, prices, soc)
            plt.show()


def get_data():
    DA = pd.read_excel(DA_PRICES_PATH)
    da_forecasted = DA['MSTL'].values
    da_actual = DA['y'].values

    IDA1 = pd.read_excel(IDA1_SCENARIOS_PATH)
    print(IDA1.columns)

    # IDA1 = pd.read_excel(IDA1_SCENARIOS_WITH_WEIGHTS_PATH)
    ida_actual = IDA1['y'].values
    ida_forecasted = IDA1['MSTL'].values
    ida_scenarios = IDA1.filter(regex='S_').T.values

    return da_forecasted, da_actual, ida_forecasted, ida_actual, ida_scenarios


def plot_cvar_vs_profit(da_forecasted, da_actual, ida_scenarios):
    res = {}
    for l in np.linspace(0, 1, 11):
        print(f"Risk Averse Factor: {l}")
        a1_schedule, adjustments, soc_scenarios, cvar, e_profit_a2 = solve_cvar(
            da_forecasted, ida_scenarios, risk_averse_factor=l)
        print("All adjustments the same:", all(
            np.allclose(adjustments[0], a) for a in adjustments))
        scenario_total_profits = [np.dot(-da_actual, a1_schedule) + np.dot(-np.array(prices), trades)
                                  for trades, prices in zip(adjustments, ida_scenarios)]

        worst_case_total_profit = min(scenario_total_profits)
        best_case_total_profit = max(scenario_total_profits)
        expected_total_profit = np.dot(-da_actual, a1_schedule) + e_profit_a2
        std_total_profits = np.std(scenario_total_profits)
        print("CVaR:", cvar)
        print("std of scenario profits:", std_total_profits)
        print("best-case total profit:", best_case_total_profit)
        print("worst-case total profit:", worst_case_total_profit)

        res[l] = (
            cvar,
            worst_case_total_profit,
            expected_total_profit,
            std_total_profits
        )
        print('\n')

    plt.plot(list(res.keys()), [v[0] for v in res.values()], label='CVaR')
    plt.plot(list(res.keys()), [v[1]
             for v in res.values()], label='Worst-case Profit')
    plt.plot(list(res.keys()), [v[2]
             for v in res.values()], label='Expected Profit')
    plt.title(f"Battery: {POWER} MW, {CAPACITY} MWh")
    plt.xlabel("Risk Averse Factor")
    plt.ylabel("Value (eur)")
    plt.axhline(y=0, color='w', linestyle='-', alpha=0.2, zorder=1)
    plt.axvline(x=0, color='w', linestyle='-', alpha=0.2, zorder=1)
    plt.legend()
    plt.show()


def main():
    da_forecasted, da_actual, ida_forecasted, ida_actual, ida_scenarios = get_data()

    if FIND_BEST_LAMBDA:
        print("========== Finding Best Lambda ==========")
        plot_cvar_vs_profit(da_forecasted, da_actual,
                            ida_scenarios)

    a1_schedule, adjustments, soc_scenarios, cvar, _ = solve_cvar(
        da_forecasted, ida_scenarios, risk_averse_factor=RISK_AVERSE_FACTOR)

    print(
        f"\n========== Solving for risk factor: {RISK_AVERSE_FACTOR} ==========")
    print("CVaR:", cvar)
    scenario_profits = [np.dot(-da_actual, a1_schedule) + np.dot(-prices, trades)
                        for trades, prices in zip(adjustments, ida_scenarios)]
    print("std of total profits:", np.std(scenario_profits))
    print("worst-case total profit:", min(scenario_profits))
    print("best-case total profit:", max(scenario_profits))

    print("\n========== Scenario Results ==========")
    if PLOT_SCENARIO_RESULTS:
        plot_scenario_results(a1_schedule, adjustments,
                            da_actual, ida_scenarios, soc_scenarios)
    expected_total_profit = np.dot(-da_actual, a1_schedule) + (
        1/len(adjustments)) * sum(np.dot(-scenario, y) for y, scenario in zip(adjustments, ida_scenarios))
    print("Expected total profit:", expected_total_profit)

    print("\n========== Final Results ==========")
    a2_schedule, soc = solve_single(ida_forecasted, a1_schedule)
    print("Actual A1 profit:", np.dot(-da_actual, a1_schedule))
    print("Actual A2 profit:", np.dot(-ida_actual, a2_schedule), "+")
    print("Actual total profit:", np.dot(-da_actual,
          a1_schedule) + np.dot(-ida_actual, a2_schedule), "=\n")
    if PLOT_FINAL_RESULTS:
        show_schedule(a1_schedule, a2_schedule, da_actual, ida_actual, soc)


def solve_cvar(da_forecasted, ida_scenarios, risk_averse_factor=1):
    model = gp.Model("Master Combined")
    model.setParam('OutputFlag', 0)  # Turn off Gurobi output
    N = len(ida_scenarios)

    # weights = [0.065625, 0.375, 0.478125, 0.08125]
    x = model.addVars(TIME, lb=-MAX_TRADE, ub=MAX_TRADE, name="da_schedule")
    y = model.addVars(TIME, N, lb=-MAX_TRADE, ub=MAX_TRADE,
                      name="ida_adjustments")
    soc = model.addVars(TIME+1, N, lb=0, ub=CAPACITY, name="soc")

    for s in range(N):
        model.addConstr(soc[0, s] == 0, name=f"initial_soc_s{s}")

    for t in range(TIME):
        for s in range(N):
            model.addConstr(
                soc[t+1, s] == soc[t, s] +
                (EFFICIENCY * x[t]) + (EFFICIENCY * y[t, s]),
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

    scenario_profit = {}
    for s in range(N):
        scenario_profit[s] = gp.quicksum(
            -ida_scenarios[s][t] * y[t, s] for t in range(TIME))

    alpha = model.addVar(lb=0, name="VaR")
    z = model.addVars(N, lb=0, name="excess_loss")
    loss = model.addVars(N, lb=0, name="loss")

    # expected_scenario_profit = gp.quicksum(
    #     weights[s] * scenario_profit[s] for s in range(N))
    expected_scenario_profit = (
        1/N)*gp.quicksum(scenario_profit[s] for s in range(N))

    for s in range(N):
        model.addConstr(
            loss[s] >= expected_scenario_profit - scenario_profit[s])
        model.addConstr(z[s] >= loss[s] - alpha)

    cvar = model.addVar(lb=0, name="cvar")

    model.addConstr(cvar >= alpha + (1 / (N * (1 - BETA)))
                    * gp.quicksum(z[s] for s in range(N)))

    expected_profit = gp.quicksum(-da_forecasted[t] * x[t]
                                  for t in range(TIME)) + (1/N)*gp.quicksum(scenario_profit[s] for s in range(N))
    #   for t in range(TIME)) + gp.quicksum(weights[s] * scenario_profit[s] for s in range(N))
    model.setObjective(expected_profit - cvar *
                       risk_averse_factor, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        # print("\n")
        # print("Expected profit:", expected_scenario_profit.getValue())
        # print("profit per scenario:", [
        #       scenario_profit[s].getValue() for s in range(N)])
        # print("Loss:", [loss[s].x for s in range(N)])
        # print("VaR:", alpha.x)
        # print("Excess loss (zi):", [z[s].x for s in range(N)])
        return (
            [x[t].x for t in range(TIME)],
            [[y[t, s].x for t in range(TIME)] for s in range(N)],
            [[soc[t+1, s].x for t in range(TIME)] for s in range(N)],
            cvar.X,
            expected_scenario_profit.getValue()
        )
    else:
        raise Exception("Master is infeasible")


# ==================== #
def solve_single(price_forecast, prev_trades):
    model = gp.Model("Master Combined")
    model.setParam('OutputFlag', 0)  # Turn off Gurobi output
    TIME = 48

    x = model.addVars(TIME, lb=-MAX_TRADE, ub=MAX_TRADE, name="da_schedule")
    soc = model.addVars(TIME+1, lb=0, ub=CAPACITY, name="soc")

    model.addConstr(soc[0] == 0, name=f"initial_soc")

    for t in range(TIME):
        model.addConstr(
            soc[t+1] == soc[t] + x[t] + prev_trades[t],
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

    profit = gp.quicksum(-(price_forecast[t] * x[t]) for t in range(TIME))

    model.setObjective(profit, gp.GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return (
            [x[t].x for t in range(TIME)],
            [soc[t+1].x for t in range(TIME)]
        )
    else:
        raise Exception("Master is infeasible")


def solve_ro(da_forecasted, ida_scenarios, max_adjustments_y):
    model = gp.Model("Master Combined")
    model.setParam('OutputFlag', 0)  # Turn off Gurobi output
    TIME = 48
    N = len(ida_scenarios)

    x_buy = model.addVars(TIME, lb=0, ub=MAX_TRADE, name="da_schedule")
    x_sell = model.addVars(TIME, lb=0, ub=MAX_TRADE, name="da_schedule")
    y_buy = model.addVars(TIME, N, lb=0, name="ida_adjustments")
    y_sell = model.addVars(TIME, N, lb=0, name="ida_adjustments")
    soc = model.addVars(TIME+1, N, lb=0, ub=CAPACITY, name="soc")
    charge = model.addVars(TIME, N, vtype=GRB.BINARY, name="charge")

    for s in range(N):
        model.addConstr(soc[0, s] == 0, name=f"initial_soc_s{s}")

    for t in range(TIME):
        for s in range(N):
            model.addConstr(
                soc[t+1, s] == soc[t, s] +
                (x_buy[t] - x_sell[t]) + (y_buy[t, s] - y_sell[t, s]),
                name=f"soc_balance_t{t}_s{s}"
            )
            model.addConstr(
                (x_buy[t] - x_sell[t]) + (y_buy[t, s] -
                                          y_sell[t, s]) <= POWER * charge[t, s],
                name=f"physical_charge_limit_t{t}_s{s}"
            )
            model.addConstr(
                (x_buy[t] - x_sell[t]) + (y_buy[t, s] - y_sell[t, s]
                                          ) >= - POWER * (1 - charge[t, s]),
                name=f"physical_discharge_limit_t{t}_s{s}"
            )
            model.addConstr(
                y_buy[t, s] <= max_adjustments_y[t],
                name=f"max_trade_x_upper_t{t}"
            )
            model.addConstr(
                y_sell[t, s] <= max_adjustments_y[t],
                name=f"max_trade_x_lower_t{t}"
            )

    z = model.addVar(name="worst_case_profit")

    for s in range(N):
        profit_da = gp.quicksum(
            da_forecasted[t] * (x_sell[t] - x_buy[t]) for t in range(TIME))
        profit_ida = gp.quicksum(
            ida_scenarios[s][t] * (y_sell[t, s] - y_buy[t, s]) for t in range(TIME))
        model.addConstr(z <= profit_da + profit_ida)

    model.setObjective(z, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        return (
            [(x_buy[t].x - x_sell[t].x) for t in range(TIME)],
            [[(y_buy[t, s].x - y_sell[t, s].x)
              for t in range(TIME)] for s in range(N)],
            [[soc[t+1, s].x for t in range(TIME)] for s in range(N)],
        )
    else:
        raise Exception("Master is infeasible")


if __name__ == "__main__":
    main()
