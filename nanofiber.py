from .utils import get_molecular_diameter
import numpy as np
import pandas as pd


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

    @property
    def info_dataframe(self):
        rows = [
            ["Sensing Material", self.sensitive_material.name],
            ["Nanofiber Width [m]", self.width],
            ["Nanofiber Length [m]", self.length],
            ["Mesh number in Width", self.n_x],
            ["Mesh number in Length", self.n_y],
            ["Surface Carrier Density", self.surface_carrier_density],
            ["Carriers Quantity", self.carriers_quantity],
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
