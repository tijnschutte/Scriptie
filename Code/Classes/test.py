

from matplotlib import pyplot as plt
import numpy as np


class StressTest():
    def __init__(self, ems, auction_data):
        self.ems = ems
        self.auction_data = auction_data
        self.tests = {
            "shock": self.run_shock_test
        }

    def run(self):
        print(f"\n{self.auction_data.current_date}")
        print("===========")

        date_mask = self.auction_data.testing_data['DA']['ds'].dt.date == self.auction_data.current_date
        actual_prices = self.auction_data.testing_data['DA'].loc[date_mask, 'y'].dropna(
        ).values
        shock_type = np.random.choice(
            ["single_fast", "single_slow_recovery", "multiple_shocks"])
        self.auction_data.testing_data['DA'].loc[date_mask, 'y'] = self.apply_shock_pattern(
            actual_prices, pattern_type=shock_type)

        stages, profit, cycles = self.ems.run()
        return profit

    def run_shock_test(self):
        print("\nRunning shock test")
        print("Access to data:", self.auction_data.testing_data != None)

        actual_prices = self.auction_data.get_auction_prices('IDA1')
        scenarios = []
        for _ in range(1000):
            sim = actual_prices.copy()
            shock_type = np.random.choice(
                ["single_fast", "single_slow_recovery", "multiple_shocks"])
            sim = self.apply_shock_pattern(sim, pattern_type=shock_type)
            scenarios.append(sim)
        plt.plot(np.array(scenarios).T)
        plt.show()
        return np.array(scenarios)

    def apply_shock_pattern(self, sim, pattern_type="single_fast"):
        print(sim)
        if pattern_type == "single_fast":
            # Apply a single sudden drop with quick recovery
            idx = np.random.randint(0, len(sim))
            shock_factor = np.clip(np.random.normal(
                0.1, 0.5), 0.1, 2)  # Sudden drop
            sim[idx] *= shock_factor
        elif pattern_type == "single_slow_recovery":
            # Apply a single sudden drop followed by gradual recovery
            # Ensure there is room for recovery
            idx = np.random.randint(0, len(sim) - 5)
            shock_factor = np.clip(np.random.normal(
                0.1, 0.5), 0.1, 2)  # Sudden drop
            sim[idx] *= shock_factor
            # Gradual recovery over 5 steps
            recovery_steps = np.linspace(shock_factor, 1.0, num=5)
            for i, factor in enumerate(recovery_steps):
                sim[idx + i] *= factor
        elif pattern_type == "multiple_shocks":
            # Apply multiple random shocks
            shock_indices = np.random.choice(
                len(sim), size=np.random.randint(2, 6), replace=False)
            for idx in shock_indices:
                shock_factor = np.clip(np.random.normal(0.5, 1.5), 0.1, 2)
                sim[idx] *= shock_factor
        return sim
