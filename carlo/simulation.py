from typing import NamedTuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .distribution import (
    get_sample_maxwell_boltzmann_velocity_distribution as get_v_dist,
)
from .environment import Environment
from .materials import Material
from .nanofiber import NanoFiber
from .utils import get_simulation_name


class SimulationSetup(NamedTuple):
    nano_fiber_width: float
    nano_fiber_length: float
    n_x: int
    n_y: int
    nano_fiber_material: Material
    temperature: float
    pressure: float
    container_volume: float
    concentration: float
    max_distance: float
    active_gas: Material
    passive_gas: Material


class GasSensorSimulation:
    def __init__(self, setup: SimulationSetup):
        self.nanofiber = NanoFiber(
            width=setup.nano_fiber_width,
            length=setup.nano_fiber_length,
            n_x=setup.n_x,
            n_y=setup.n_y,
            material=setup.nano_fiber_material,
        )
        self.environment = Environment(
            temperature=setup.temperature,
            pressure=setup.pressure,
            container_volume=setup.container_volume,
            concentration=setup.concentration,
            max_distance=setup.max_distance,
            active_gas=setup.active_gas,
            passive_gas=setup.passive_gas,
        )

        self.state_mtx = np.ones(self.nanofiber.grid_size, dtype=np.bool_)
        self.time_mtx = np.zeros(self.nanofiber.grid_size)
        self.iterations = 0

        self.active_v_dist = get_v_dist(
            self.environment.temperature,
            self.environment.active_gas_molweight,
        )
        self.passive_v_dist = get_v_dist(
            self.environment.temperature,
            self.environment.passive_gas_molweight,
        )
        if (
            self.environment.active_gas_quantity / self.nanofiber.carriers_quantity
        ) < 1:
            raise Exception("Active gas shortage. Check input parameters.")

    @property
    def maximum_current(self):
        Q_E = 1.6021e-19
        return self.nanofiber.carriers_quantity * Q_E

    def get_gasses_probabilities(self):
        p_active_gas = self.environment.concentration
        p_passive_gas = 1 - (self.environment.concentration)
        return p_passive_gas, p_active_gas

    def get_mixture_mtx(self):
        binary_values = np.array([True, False], dtype=np.bool_)
        p = self.get_gasses_probabilities()
        return np.random.choice(binary_values, self.nanofiber.grid_size, p=p)

    def update_time_mtx(self, mixture_mtx):
        v_active_mtx = np.random.choice(
            self.active_v_dist[0, :],
            self.nanofiber.grid_size,
            p=self.active_v_dist[1, :],
        )
        v_passive_mtx = np.random.choice(
            self.passive_v_dist[0, :],
            self.nanofiber.grid_size,
            p=self.passive_v_dist[1, :],
        )

        v_mtx = np.where(mixture_mtx == 0, v_active_mtx, v_passive_mtx)
        distance_mtx = (
            np.random.rand(*self.nanofiber.grid_size) * self.environment.max_distance
        )
        self.time_mtx += (distance_mtx / v_mtx) * self.state_mtx

    @property
    def active_cell_ratio(self):
        total_cells = self.nanofiber.n_x * self.nanofiber.n_y
        active_cells = self.state_mtx.sum()
        active_ratio = active_cells / total_cells
        return active_ratio

    def is_converged(self, active_cell_ratio_to_converge):
        return self.active_cell_ratio < active_cell_ratio_to_converge

    def run(self, active_cell_ratio_to_converge):
        total_remained = self.active_cell_ratio - active_cell_ratio_to_converge
        pbar = tqdm(total=100, desc="Simulating")
        while not self.is_converged(active_cell_ratio_to_converge):
            mixture_mtx = self.get_mixture_mtx()
            self.state_mtx *= mixture_mtx
            self.update_time_mtx(mixture_mtx)
            self.iterations += 1
            self.simulation_time = (self.time_mtx * -1 * (self.state_mtx - 1)).max()
            remained = self.active_cell_ratio - active_cell_ratio_to_converge
            advance = int((total_remained - remained) * 100 / total_remained)
            postfix = {"ACR": self.active_cell_ratio, "ITR": self.iterations}
            pbar.set_postfix(postfix)
            pbar.update(max(advance - pbar.n - 1, 0))
        pbar.close()

    @property
    def info_dataframe(self):
        rows = [
            ["Sensing Material", self.nanofiber.material.name],
            ["Active Gas", self.environment.active_gas.name],
            ["Passive Gas", self.environment.passive_gas.name],
            ["Iteration Number", self.iterations],
            ["Active Cells Ratio", self.active_cell_ratio],
            ["Active Gas Concentration", self.environment.concentration],
            ["Nanofiber Width [m]", self.nanofiber.width],
            ["Nanofiber Length [m]", self.nanofiber.length],
            ["Mesh number in Width", self.nanofiber.n_x],
            ["Mesh number in Length", self.nanofiber.n_y],
            ["Container Pressure [Pa]", self.environment.pressure],
            ["Container Temperature [K]", self.environment.temperature],
            ["Container Volume [m^3]", self.environment.container_volume],
            ["Max Distance [m]", self.environment.max_distance],
        ]
        return pd.DataFrame(rows, columns=["Property", "Description"])

    def __repr__(self):
        return self.info_dataframe.to_string(
            formatters={"Property": "{:<30}".format, "Description": "{:<80}".format},
            float_format="{:<80}".format,
            justify="left",
            index=False,
            header=False,
        )

    def __str__(self):
        return self.__repr__()
