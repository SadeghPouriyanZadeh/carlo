import numpy as np
from rich.progress import Progress
from .nanofiber import NanoFiber
from .environment import Environment
from .distribution import (
    get_sample_maxwell_boltzmann_velocity_distribution as get_v_dist,
)


class GasSensorSimulation:
    def __init__(
        self, nanofiber: NanoFiber, environment: Environment,
    ):
        self.nanofiber = nanofiber
        self.environment = environment

        self.state_mtx = np.ones(self.nanofiber.grid_size, dtype=np.bool_)
        self.time_mtx = np.zeros(self.nanofiber.grid_size)
        self.iterations = 0

        self.active_v_dist = get_v_dist(
            self.environment.temperature, self.environment.active_gas_molweight
        )
        self.passive_v_dist = get_v_dist(
            self.environment.temperature, self.environment.passive_gas_molweight
        )
        if (
            self.environment.active_gas_quantity / self.nanofiber.carriers_quantity
        ) < 1:
            raise ValueError("Active gas shortage. Check input parameters.")

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
        with Progress() as progress:
            task = progress.add_task("[red]Solving...", total=100)
            while not self.is_converged(active_cell_ratio_to_converge):
                mixture_mtx = self.get_mixture_mtx()
                self.state_mtx *= mixture_mtx
                #                 self.update_time_mtx(mixture_mtx)
                self.update_time_mtx(mixture_mtx)
                self.iterations += 1
                self.simulation_time = (self.time_mtx * -1 * (self.state_mtx - 1)).max()
                remained = self.active_cell_ratio - active_cell_ratio_to_converge
                advance = int((total_remained - remained) * 100 / total_remained)
                description_parts = [
                    f"ACR: {self.active_cell_ratio:.2e}",
                    f"ITR: {self.iterations:07}",
                ]
                description = "{" + ", ".join(description_parts) + "}"
                progress.update(task, completed=advance, description=description)
            progress.update(task, completed=100, description="Solution Converged")
