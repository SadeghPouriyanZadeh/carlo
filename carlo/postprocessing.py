import numpy as np
from .simulation import GasSensorSimulation
from .utils import get_current_time_function
import matplotlib.pyplot as plt


class CurrentEstimator:
    def __init__(self, postprocess_obj):
        self.postprocess_obj = postprocess_obj
        self.a = self.postprocess_obj.simulation_obj.maximum_current
        self.b = self.calculate_b()

    def calculate_b(self):
        time_current_array = self.postprocess_obj.get_time_current_array()
        index = -1
        t, i = time_current_array[:, index]
        b = (-1 / t) * np.log((self.a - i) / self.a)
        return b

    def __call__(self, time):
        return self.a * (1 - np.exp(-self.b * time))


class PostProcess:
    def __init__(self, simulation_obj: GasSensorSimulation):
        self.simulation_obj = simulation_obj
        self.current_estimator = CurrentEstimator(self)

    def get_state_generator(self):
        flat_time_mtx = np.unique(self.simulation_obj.time_mtx.flatten())
        for t in flat_time_mtx:
            step_state = np.where(
                self.simulation_obj.time_mtx <= t, self.simulation_obj.state_mtx, 1
            )
            yield t, step_state

    def get_current_from_state(self, state):
        Q_E = 1.6021e-19
        n_e = (
            self.simulation_obj.nanofiber.n_y * self.simulation_obj.nanofiber.n_x
        ) - state.sum()
        current = n_e * Q_E * self.simulation_obj.nanofiber.cell_carriers
        return current

    def get_time_current_array(self):
        state_generator = self.get_state_generator()
        time_list = []
        current_list = []
        for t, step_state in state_generator:
            time_list.append(t)
            current = self.get_current_from_state(step_state)
            current_list.append(current)
        return np.array([np.array(time_list), np.array(current_list)])

    def get_current_figure(self, datapoints=10):
        time_current_array = self.get_time_current_array()
        interval = int(
            (self.simulation_obj.nanofiber.n_x * self.simulation_obj.nanofiber.n_y)
            / datapoints
        )
        t = time_current_array[0, ::interval]
        y_hat = self.current_estimator(t)
        y = time_current_array[1, ::interval]

        plt.figure(figsize=(5, 3), dpi=100, facecolor="white")
        plt.scatter(t, y * 1e9, color="red", label="Simulation Data")
        plt.plot(t, y_hat * 1e9, color="blue", label="Fitted Curve")
        plt.xlabel("t $[s]$")
        plt.ylabel("I $[nA]$")
        plt.legend(frameon=False)
        fig = plt.gcf()
        return fig
