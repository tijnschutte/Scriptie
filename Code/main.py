import json
import pandas as pd
from Classes.battery import Battery
from Classes.data import AuctionData
from Classes.ems import StochasticEMS, DeterministicEMS
from Classes.simulate import Simulation
from config import *
import matplotlib.pyplot as plt
plt.rcParams.update({
    'figure.figsize': (8, 6),        # Default figure size
    'font.size': 12,                # Font size for labels and titles
    'axes.titlesize': 14,           # Title font size
    'axes.labelsize': 12,           # Axis label font size
    'xtick.labelsize': 10,          # X-axis tick size
    'ytick.labelsize': 10,          # Y-axis tick size
    'axes.grid': True,              # Add grid by default
    'grid.alpha': 0.5,              # Make grid lines subtle
    'grid.linestyle': '--',         # Dashed grid lines
    'legend.fontsize': 10,          # Legend font size
    'legend.frameon': True,         # Box around the legend
    'legend.loc': 'best',           # Best location for the legend
    'savefig.dpi': 300,             # High-resolution saves
    'savefig.format': 'png',        # Default save format
    'lines.linewidth': 1.5,           # Thicker lines
    'lines.markersize': 6,          # Marker size
})
plt.rcParams['font.family'] = 'serif'
pd.options.display.float_format = '{:.2f}'.format


def main():
    auction_data = AuctionData()

    battery = Battery(POWER, CAPACITY, MAX_TRADE, EFFICIENCY)
    ems = StochasticEMS(battery, auction_data, RISK_AVERSE_FACTOR) if RISK_AVERSE_FACTOR != None else DeterministicEMS(
        battery, auction_data)

    sim = Simulation(ems, auction_data)
    res = sim.run(auction_data.end_date)

    file = f'./Results/{YEAR}/TEST:results_battery={battery.capacity}MWh;{battery.power}MW;{BETA*100}%;l{RISK_AVERSE_FACTOR}.json'
    with open(file, 'w') as f:
        json.dump(res, f)


if __name__ == '__main__':
    main()
