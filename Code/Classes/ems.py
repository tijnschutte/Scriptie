
from statsforecast.models import MSTL, AutoARIMA
from statsforecast import StatsForecast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import *


class EMS:
    def __init__(self, battery, auction_data):
        self.auction_data = auction_data
        self.battery = battery

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
        prices = [
            self.auction_data.prediction_day[f'y_{auction}'] for auction in schedules.keys()]
        _, (price_ax, power_ax) = plt.subplots(2, 1)

        schedule = pd.DataFrame()
        for auction, bids in schedules.items():
            schedule[auction] = bids
        schedule.index = self.auction_data.prediction_day['ds']
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

    def calculate_profit(self, schedules):
        profit = 0
        for auction, schedule in schedules.items():
            date_mask = self.auction_data.testing_data[auction]['ds'].dt.date == self.auction_data.current_date
            prices = self.auction_data.testing_data[auction].loc[date_mask, 'y'].fillna(
                0).values
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
        }

    def rate_forecast(self, auction, forecast):
        date_mask = self.auction_data.testing_data[auction]['ds'].dt.date == self.auction_data.current_date
        actual = self.auction_data.testing_data[auction].loc[date_mask, 'y'].dropna(
        ).values
        nmae = np.sum(np.abs(actual - forecast)) / np.sum(np.abs(actual)) * 100
        return nmae
