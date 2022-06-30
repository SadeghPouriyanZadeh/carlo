from .utils import get_molecular_diameter
import numpy as np
import pandas as pd
from .materials import Material
from typing import Tuple


def get_surface_carrier_density(material: Material) -> float:
    diameter = get_molecular_diameter(
        material.density,
        material.molweight,
    )
    area = np.pi * (diameter**2) / 4
    return 1 / area


def get_cell_carriers(
    width: float,
    length: float,
    surface_carrier_density: float,
    grid_size: Tuple[int, int],
) -> int:
    n_y, n_x = grid_size
    area = width * length
    num = surface_carrier_density * area
    den = n_x * n_y
    return num // den


class NanoFiber:
    def __init__(
        self,
        width: float,
        length: float,
        n_x: int,
        n_y: int,
        material: Material,
    ) -> None:
        self.width = width
        self.length = length
        self.n_x = n_x
        self.n_y = n_y
        self.material = material
        self.surface_carrier_density = get_surface_carrier_density(material)
        self.grid_size = (self.n_y, self.n_x)
        self.cell_carriers = get_cell_carriers(
            width,
            length,
            self.surface_carrier_density,
            self.grid_size,
        )
        self.carriers_quantity = (width * length) * self.surface_carrier_density

    @property
    def info_dataframe(self) -> pd.DataFrame:
        rows = [
            ["Sensing Material", self.material.name],
            ["Nanofiber Width [m]", self.width],
            ["Nanofiber Length [m]", self.length],
            ["Mesh number in Width", self.n_x],
            ["Mesh number in Length", self.n_y],
            ["Surface Carrier Density", self.surface_carrier_density],
            ["Carriers Quantity", self.carriers_quantity],
        ]
        return pd.DataFrame(rows, columns=["Property", "Description"])

    def __repr__(self) -> str:
        return self.info_dataframe.to_string(
            formatters={"Property": "{:<30}".format, "Description": "{:<80}".format},
            float_format="{:<80}".format,
            justify="left",
            index=False,
            header=False,
        )

    def __str__(self) -> str:
        return self.__repr__()
