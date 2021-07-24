import numpy as np


def get_molecular_diameter(solid_state_density, molweight):
    NA = 6.0221409e23  # avogadro's number
    PI = np.pi

    n_mol = solid_state_density / molweight  # particles moles
    n_particles = n_mol * NA
    particle_volume = 1 / n_particles
    particle_diameter = ((6 * particle_volume) / PI) ** (1 / 3)
    return particle_diameter
