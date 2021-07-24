from .utils import get_molecular_diameter
import numpy as np


class NanoFiber:
    def __init__(self, width, length, n_x, n_y, sensitive_material):
        self.width = width
        self.length = length
        self.n_x = n_x
        self.n_y = n_y
        self.sensitive_material = sensitive_material
        self.surface_carrier_density = self.get_surface_carrier_density()
        self.grid_size = (self.n_y, self.n_x)
        self.cell_carriers = self.get_cell_carriers()
        self.carriers_quantity = (width * length) * self.surface_carrier_density

    def get_surface_carrier_density(self):
        PI = np.pi
        diameter = get_molecular_diameter(
            self.sensitive_material.density, self.sensitive_material.molweight
        )
        area = PI * (diameter ** 2) / 4
        return 1 / area

    def get_cell_carriers(self):
        num = self.surface_carrier_density * self.length * self.width
        den = self.n_x * self.n_y
        return int(num / den)
