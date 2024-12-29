import pandas as pd
from config import *


class Simulation:
    def __init__(self, ems, auction_data):
        self.ems = ems
        self.auction_data = auction_data
        self.results = []

    def run(self, end_date):

        results = {}

        print(f"Running simulation until {end_date}")
        while self.auction_data.current_date <= end_date:
            print(f"\n{self.auction_data.current_date}")
            print("===========")

            if CVAR_PLOT_DAY != None and self.auction_data.current_date == pd.to_datetime(CVAR_PLOT_DAY).date():
                self.ems.plot_cvars()

            stages, profit, cycles = self.ems.run()

            results[self.auction_data.current_date.strftime('%Y-%m-%d')] = {
                'profit': profit,
                'auctions': stages,
                'cycles': cycles
            }

            self.auction_data.move_to_next_day()
            print(
                f"Current total profit: {sum([v['profit'] for v in results.values()])}")

        print(f"Total profit: {sum([v['profit'] for v in results.values()])}")
        return results

    def summarize_results(self):
        pass
