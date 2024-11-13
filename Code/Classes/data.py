import pandas as pd


class AuctionData:
    def __init__(self):
        df = pd.read_excel('./Data/market_data.xlsx')
        self.da = pd.DataFrame({
            'ds': df['Date'],
            'unique_id': 1,
            'y': df['IE DA EUR']
        })
        self.ida1 = pd.DataFrame({
            'ds': df['Date'],
            'unique_id': 1,
            'y': df['IE IDA1 EUR price']
        })
        self.ida2 = pd.DataFrame({
            'ds': df['Date'],
            'unique_id': 1,
            'y': df['IE IDA2 EUR price']
        })
        self.cross_vals = {
            'IDA1': pd.read_excel('./Data/Crossvalidation/cross_val_ida1.xlsx'),
            'IDA2': pd.read_excel('./Data/Crossvalidation/cross_val_ida2.xlsx')
        }
        self.__clean_data()
        self.__split_data()

    def __clean_data(self):
        self.da = self.da.resample(
            '30min', on='ds').mean().interpolate().reset_index()
        self.da['ds'] = self.da['ds'].dt.floor('s')

        self.ida1 = self.ida1.resample(
            '30min', on='ds').mean().interpolate().reset_index()
        self.ida1['ds'] = self.ida1['ds'].dt.floor('s')

        self.ida2['ds'] = self.ida2['ds'].dt.floor('s')

    def __split_data(self):
        train_ida1, test_ida1 = self.ida1[self.ida1['ds'].dt.year <
                                          2023], self.ida1[self.ida1['ds'].dt.year == 2023]
        train_ida2, test_ida2 = self.ida2[self.ida2['ds'].dt.year <
                                          2023], self.ida2[self.ida2['ds'].dt.year == 2023]
        train_da, test_da = self.da[self.da['ds'].dt.year <
                                    2023], self.da[self.da['ds'].dt.year == 2023]
        self.current_date = test_da['ds'].iloc[0].date()
        self.training_data = {'DA': train_da,
                              'IDA1': train_ida1,
                              'IDA2': train_ida2
                              }
        self.testing_data = {'DA': test_da,
                             'IDA1': test_ida1,
                             'IDA2': test_ida2
                             }
        self.max_sim_length = len(self.testing_data['DA'])
        self.prediction_day = pd.DataFrame({
            'ds': pd.date_range(start=self.current_date, periods=48, freq='30min'),
            'unique_id': 1,
        })

    def move_to_next_day(self):
        self.update_past_errors()
        self.current_date += pd.Timedelta(days=1)
        self.training_data['DA'] = self.da[self.da['ds'].dt.date <
                                           self.current_date]
        self.training_data['IDA1'] = self.ida1[self.ida1['ds'].dt.date <
                                               self.current_date]
        self.training_data['IDA2'] = self.ida2[self.ida2['ds'].dt.date <
                                               self.current_date]
        self.testing_data['DA'] = self.da[self.da['ds'].dt.date >=
                                          self.current_date]
        self.testing_data['IDA1'] = self.ida1[self.ida1['ds'].dt.date >=
                                              self.current_date]
        self.testing_data['IDA2'] = self.ida2[self.ida2['ds'].dt.date >=
                                              self.current_date]
        self.prediction_day = pd.DataFrame({
            'ds': pd.date_range(start=self.current_date, periods=48, freq='30min'),
            'unique_id': 1,
        })

    def update_past_errors(self):
        exos = {'IDA1': [], 'IDA2': ['y_DA']}
        for auction in exos.keys():
            cross_val = self.cross_vals[auction]
            y = self.get_auction_prices(auction)
            # fixme: cross-val still needs to be done with auctions as second stages
            forecast = self.get_forecast_from_file(auction)
            trading_hours = cross_val['ds'].dt.hour.unique()
            timeframe_mask = self.prediction_day['ds'].dt.floor(
                'h').dt.hour.isin(trading_hours)
            df = pd.DataFrame({
                'ds': self.prediction_day['ds'].loc[timeframe_mask],
                'y': y,
                'MSTL': forecast,
                'error': y - forecast
            })
            self.cross_vals[auction] = pd.concat(
                [cross_val, df]).reset_index(drop=True)
            print(f"Cross-validation updated for {auction}\n")

    def get_auction_prices(self, auction):
        date_mask = self.testing_data[auction]['ds'].dt.date == self.current_date
        return self.testing_data[auction].loc[date_mask, 'y'].dropna().values

    def get_forecast_from_file(self, auction, given_exos=None):
        exo_vars = given_exos if given_exos is not None else self.training_data[auction].drop(
            columns=['y', 'ds', 'unique_id']).columns.tolist()
        file = f'./Data/Forecasts/{auction}/{self.current_date}_exo{exo_vars}.xlsx'
        print(f"getting forecast from {file}")
        forecast = pd.read_excel(file)
        return forecast['MSTL'].values

    def get_trading_window(self, auction):
        start_time = self.training_data[auction].dropna()[
            'ds'].dt.hour.min() * 2
        end_time = (self.training_data[auction].dropna()[
            'ds'].dt.hour.max() + 1) * 2
        return start_time, end_time

    def realize_prices(self, auction):
        date_mask = self.testing_data[auction]['ds'].dt.date == self.current_date
        auction_prices = self.testing_data[auction].loc[date_mask, 'y'].values
        self.prediction_day[f'y_{auction}'] = auction_prices

        auctions = ['DA', 'IDA1', 'IDA2']
        for key in auctions[auctions.index(auction)+1:]:
            print(f"Adding {auction} prices to {key} training data")
            self.training_data[key] = self.training_data[key].copy()
            merged_data = pd.merge(self.training_data[key],
                                   self.training_data[auction][[
                                       'ds', 'y']],
                                   on='ds',
                                   how='left',  # Keep all original rows for key
                                   suffixes=('', f'_{auction}'))
            self.training_data[key] = merged_data
