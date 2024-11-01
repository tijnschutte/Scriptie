import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from Classes.battery import Battery
from Classes.data import AuctionData
from Classes.sEMS import StochasticEMS
from Classes.simulate import Simulation
from config import *
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


def main():
    battery = Battery(POWER, CAPACITY, MAX_TRADE, EFFICIENCY)
    auction_data = AuctionData()
    ems = StochasticEMS(battery, auction_data)
    sim = Simulation(ems, auction_data)

    res = sim.run(31)
    pretty_res = json.dumps(res, indent=4)
    print(pretty_res)


if __name__ == '__main__':
    main()
