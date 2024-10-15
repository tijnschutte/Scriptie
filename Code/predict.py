from matplotlib import pyplot as plt
import numpy as np
from statsforecast import StatsForecast
from statsforecast.models import MSTL, AutoARIMA
import pandas as pd
from model import solve_cvar, solve_single, plot_cvar_vs_profit, show_schedule

HISTORICAL_DATA_PATH = 'Data/IDA & DA Ierland 2023.xlsx'
HORIZON = 48
RISK_AVERSE_FACTOR = 1


def update(a2_test, a1_realized_prices):
    a2_test['a1_y'] = pd.Series(a1_realized_prices)
    return a2_test


def simulate_day():
    print("Getting Data")
    da, ida1, ida2 = get_historical_data()
    train_da, test_da, train_ida1, test_ida1, train_ida2, test_ida2 = split_data(
        da, ida1, ida2)

    print("Computing past errors")
    # ida1_error = cross_val(n_windows=10, train=train_ida1)
    ida1_error = pd.read_excel('Data/crossvalidation_errors.xlsx')
    mean_past_errors, std_past_errors = ida1_error['error'].mean(
    ), ida1_error['error'].std()**2

    current = {"history": train_da,
               "horizon": test_da[['ds', 'unique_id']].head(HORIZON),
               "real": test_da[['ds', 'y']]}
    next = {"history": train_ida1,
            "horizon": test_ida1[['ds', 'unique_id']].head(HORIZON),
            "real": test_ida1[['ds', 'y']]}
    remaining = []
    n_scenarios_next = 5
    bids = [0]*HORIZON
    profits = []
    while next:
        print("forecasting current")
        current_forecast = forecast(current['history'], current['horizon'])
        print("forecasting next")
        next_forecast = forecast(next['history'], next['horizon'])
        print("generating scenarios for next")
        scenarios_next = generate_scenarios(
            n_scenarios_next, next_forecast, mean_past_errors, std_past_errors)

        plot_cvar_vs_profit(
            current_forecast, current['real']['y'].head(HORIZON), scenarios_next)

        print("optimizing current")
        current_schedule, next_adjustments, soc_scenarios, cvar, e_scenario_profit = solve_cvar(
            current_forecast, scenarios_next, risk_averse_factor=RISK_AVERSE_FACTOR)
        bids = np.add(bids, current_schedule)
        profits = np.append(profits, np.dot(
            current_schedule, -current['real']['y'].head(HORIZON)))

        print("updating next with realized prices from current")
        next['history'] = pd.merge(next['history'], current['history'][[
            'ds', 'y']].rename(columns={'y': 'y_da'}).head(HORIZON), on='ds')
        next['horizon'] = pd.merge(next['horizon'], current['real'][[
            'ds', 'y']].rename(columns={'y': 'y_da'}).head(HORIZON), on='ds')

        print("shifting: next -> current, remaining -> next")
        current = next
        next = remaining.pop(0) if remaining else None

    print("Profit: ", profits.sum())
    current_forecast = forecast(current['history'], current['horizon'])
    current_schedule, soc = solve_single(current_forecast, bids)
    print(profits.sum() + np.dot(current_schedule, -
          test_ida1.head(HORIZON)['y'].values))
    show_schedule(bids, current_schedule, test_da.head(HORIZON)[
                  'y'].values, test_ida1.head(HORIZON)['y'].values, soc)


def optimize_ida(train_ida1, current_day_ida1, a1_schedule):

    print("Updating forecast for IDA-1 with realized DA prices")
    # forecast IDA-1 (with exogenous variables)
    ida_updated_forecast = forecast(
        train_ida1, current_day_ida1[['ds', 'unique_id', 'y_da']])

    print("Optimizing IDA-1 schedule")
    a2_schedule, soc = solve_single(ida_updated_forecast, a1_schedule)

    return a2_schedule


def optimize_da(train_da, train_ida1):
    print("Forecasting DA")
    # forecast DA (no exogenous variables)
    da_forecast = forecast(train_da)

    print("Computing past errors")
    # cross validation IDA-1
    # ida1_error = cross_val(n_windows=10, train=train_ida1)
    ida1_error = pd.read_excel('Data/crossvalidation_errors.xlsx')
    mean_error, std_error = ida1_error['error'].mean(
    ), ida1_error['error'].std()

    print("Forecasting IDA-1")
    # forecast IDA-1 (no exogenous variables)
    ida_forecast = forecast(train_ida1)

    print("Generating Scenarios for IDA-1")
    # generate scenarios for IDA-1 based on prediction and previous errors
    n_scenarios_ida1 = 10
    scenarios_ida1 = generate_scenarios(
        n_scenarios_ida1, ida_forecast, mean_error, std_error)

    print("Optimizing DA schedule")
    # get DA schedule
    a1_schedule, adjustments, soc_scenarios, cvar, e_scenario_profit = solve_cvar(
        da_forecast, scenarios_ida1, risk_averse_factor=RISK_AVERSE_FACTOR)

    return a1_schedule


def get_historical_data():
    df = pd.read_excel(HISTORICAL_DATA_PATH)
    da = pd.DataFrame({
        'ds': df['Datetime'],
        'unique_id': 1,
        'y': df['IE DA EUR']
    })
    ida1 = pd.DataFrame({
        'ds': df['Datetime'],
        'unique_id': 1,
        'y': df['IE IDA1 EUR price']
    })
    ida2 = pd.DataFrame({
        'ds': df['Datetime'],
        'unique_id': 1,
        'y': df['IE IDA2 EUR price']
    })
    return da, ida1, ida2


def split_data(da, ida1, ida2):
    train_ida1, test_ida1 = ida1[ida1['ds'].dt.month <
                                 12], ida1[ida1['ds'].dt.month >= 12]
    train_ida2, test_ida2 = ida2[ida2['ds'].dt.month <
                                 12], ida2[ida2['ds'].dt.month >= 12]
    train_da,   test_da = da[da['ds'].dt.month <
                             12],     da[da['ds'].dt.month >= 12]
    return train_da, test_da, train_ida1, test_ida1, train_ida2, test_ida2


def create_MSTL(seasons=[2, 4, 6, 8, 10, 48]):
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


def add_exogenous(df, exogenous):
    return df.merge(exogenous, on='ds')


def cross_val(n_windows, train):
    sf = create_MSTL([HORIZON, 2*HORIZON, 5*HORIZON, 6*HORIZON])

    cross_val_df = sf.cross_validation(df=train,
                                       h=HORIZON,
                                       step_size=HORIZON,
                                       n_windows=n_windows)
    cross_val_df['error'] = cross_val_df['y'] - cross_val_df['MSTL']

    return cross_val_df['error']


def forecast(train, test=None):
    model = create_MSTL([HORIZON, 2*HORIZON, 5*HORIZON, 6*HORIZON])
    predictions = model.forecast(df=train, h=HORIZON, X_df=test)
    return predictions['MSTL'].values


def generate_scenarios(n_scenarios, ida_forecast, mean_error, std_error):

    random_noise = np.random.normal(
        mean_error, std_error, (n_scenarios, HORIZON))

    simulated_scenarios = np.add(ida_forecast, random_noise)

    scenarios = pd.DataFrame(simulated_scenarios).T

    def rolling(list):
        return pd.Series(list).rolling(3).mean().fillna(ida_forecast[0])

    scenarios = scenarios.apply(rolling, axis=0)
    return scenarios.T.values.tolist()


simulate_day()
