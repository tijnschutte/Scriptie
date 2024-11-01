from config import *


class Battery:
    def __init__(self, power, capacity, max_trade, efficiency):
        self.power = power
        self.capacity = capacity
        self.max_trade = max_trade
        self.efficiency = efficiency
