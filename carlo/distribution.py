import numpy as np


def get_maxwell_boltzmann_velocity_distribution(v, temperature, molecule_mass):
    PI = np.pi
    K_B = 1.380649e-23  # boltzmann constant
    t = temperature
    m = molecule_mass
    expr1 = 4 * PI * (v ** 2)
    expr2 = (m / (2 * PI * K_B * t)) ** 1.5
    expr3 = np.exp((-m * (v ** 2)) / (2 * K_B * t))
    return expr1 * expr2 * expr3


def get_sample_maxwell_boltzmann_velocity_distribution(temperature, molar_mass):
    R = 8.314  # universal gas constant
    NA = 6.0221409e23  # avogadro's number
    molecule_mass = molar_mass / NA
    mean = np.sqrt((2 * R * temperature) / molar_mass)
    distribution = np.empty((2, 100))
    distribution[0, :] = np.linspace(0, 3 * mean, 100)
    distribution[1, :] = get_maxwell_boltzmann_velocity_distribution(
        distribution[0, :], temperature, molecule_mass
    )
    distribution[1, :] = (1 / distribution[1, :].sum()) * distribution[1, :]
    return distribution
