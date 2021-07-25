import numpy as np
from .simulation import GasSensorSimulation

class PostProcess:
    def __init__(self, simulation_obj: GasSensorSimulation):
        self.simulation_obj = simulation_obj

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
