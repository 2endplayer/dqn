import numpy as np
import gym
from gym import spaces
from gym.envs.registration import register

# ENVIRONMENT

class Godzilla(gym.Env):

    def generate_market_prices(self, num_prices, low=0.1, high=10.0):
        # Generate a random array of prices between `low` and `high`
        price = np.random.uniform(low=low, high=high, size=num_prices)

        # Round the prices to 2 decimal places
        price = np.round(price, 2)
        return price

    def calculate_profitability(self, candle_data):
        holding = False
        buy_price = 0
        total_percentage = 0

        for candle in candle_data:
            signal = int(candle[1])
            close_price = float(candle[0])

            if signal == 1 and not holding:
                buy_price = close_price
                holding = True
            elif signal == 2 and holding:
                sell_price = close_price
                holding = False
                profit = sell_price - buy_price
                profit_percentage = (profit / buy_price) * 100
                total_percentage += profit_percentage

            # If signal is 0 or holding is True (i.e., already bought but waiting to sell), do nothing

        return total_percentage

    def find_extrema(self, prices, delta, window):
        extrema_points = []
        for i in range(window, len(prices) - window):
            local_max = max(prices[i - window:i + window + 1])
            local_min = min(prices[i - window:i + window + 1])
            if prices[i] >= local_max - delta:
                extrema_points.append([i, prices[i], 2])  # high point
            elif prices[i] <= local_min + delta:
                extrema_points.append([i, prices[i], 1])  # low point
            else:
                extrema_points.append([i, prices[i], 0])  # no action

        return extrema_points

    def __init__(self):
        # price
        self.action = None
        self.prices = self.generate_market_prices(100, 0.1, 10.0)

        # action space
        self.action_space = spaces.Discrete(2)

        delta = np.array([0.0, 10])
        window = np.array([1, 12])

        self.observation_space = spaces.Box(delta, window, dtype=np.float32)

        self.last_state = 0
        self.current_step = 0
        self.max_steps = 1000
        self.state = None

    def step(self, action):

        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        # here action affect the state so profitability and get reward and if not end, so it continues
        # action should be array with delta and window
        self.action = action

        delta = action[0]
        window = action[1]

        database = self.find_extrema(self.prices, delta, window)
        profit = self.calculate_profitability(database)

        self.state = profit

        if self.state > self.last_state:
            self.last_state = self.state

        terminated = bool(self.state < self.last_state or self.state <= 0 or self.current_step >= self.max_steps)
        if not terminated:
            reward = 1.0
            self.current_step += 1
        else:
            reward = 0.0

        return np.array(self.state, dtype=np.float32), reward, terminated, False, {}

    def reset(self, **kwargs):

        self.state = 0
        self.current_step = 0

        return np.array(self.state, dtype=np.float32), {}

    def render(self, mode='human'):
        print(f"params: {self.action}, profit: {self.state}")

gym.register(
    id='Godzilla-v0',
    entry_point='gym_examples/my_env_module:Godzilla'
)
