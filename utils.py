import numpy as np
from datetime import datetime
import pickle
from pathlib import Path


def get_molecular_diameter(solid_state_density, molweight):
    NA = 6.0221409e23  # avogadro's number
    PI = np.pi

    n_mol = solid_state_density / molweight  # particles moles
    n_particles = n_mol * NA
    particle_volume = 1 / n_particles
    particle_diameter = ((6 * particle_volume) / PI) ** (1 / 3)
    return particle_diameter


def get_simulation_name():
    now = datetime.now()
    name_parts = now.year, now.month, now.day, now.hour, now.minute
    name = "_".join([str(p) for p in name_parts]) + ".carlo"
    return name


def save_simulation(simulation_obj, file_path=None):
    if not file_path:
        file_path = Path(get_simulation_name())
    with open(file_path, "wb") as file:
        pickle.dump(simulation_obj, file)
    return file_path


def load_simulation(file_path):
    with open(file_path, "rb") as file:
        return pickle.load(file)


def get_current_time_function(time_current_array, maximum_current):
    index = -1
    t, i = time_current_array[:, index]
    b = (-1 / t) * np.log((maximum_current - i) / maximum_current)

    def get_current(time):
        return maximum_current * (1 - np.exp(-b * time))

    return get_current
