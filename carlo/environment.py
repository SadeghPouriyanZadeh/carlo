import pandas as pd

from .materials import Material
from .utils import get_molecular_diameter


def get_passive_gas_quantity(
    pressure: float,
    temperature: float,
    volume: float,
) -> float:
    R = 8.314  # universal gas constant
    NA = 6.0221409e23  # avogadro's number
    num = NA * pressure * volume
    den = R * temperature
    return num / den


class Environment:
    def __init__(
        self,
        temperature: float,
        pressure: float,
        container_volume: float,
        concentration: float,
        max_distance: float,
        active_gas: Material,
        passive_gas: Material,
    ) -> None:
        self.concentration = concentration
        self.temperature = temperature
        self.pressure = pressure
        self.max_distance = max_distance
        self.active_gas = active_gas
        self.passive_gas = passive_gas
        self.active_gas_molweight = active_gas.molweight
        self.passive_gas_molweight = passive_gas.molweight
        self.active_gas_diameter = get_molecular_diameter(
            active_gas.density, active_gas.molweight
        )
        self.passive_gas_diameter = get_molecular_diameter(
            passive_gas.density, passive_gas.molweight
        )
        self.container_volume = container_volume
        self.passive_gas_quantity = get_passive_gas_quantity(
            pressure,
            temperature,
            container_volume,
        )
        self.active_gas_quantity = self.passive_gas_quantity * concentration

    @property
    def info_dataframe(self) -> pd.DataFrame:
        rows = [
            ["Active Gas", self.active_gas.name],
            ["Passive Gas", self.passive_gas.name],
            ["Active Gas Concentration", self.concentration],
            ["Container Pressure [Pa]", self.pressure],
            ["Container Temperature [K]", self.temperature],
            ["Container Volume [m^3]", self.container_volume],
            ["Max Distance [m]", self.max_distance],
            ["Active Gas Diameter [m]", self.active_gas_diameter],
            ["Passive Gas Diameter [m]", self.passive_gas_diameter],
            ["Active Gas Quantity", self.active_gas_quantity],
            ["Passive Gas Quantity", self.passive_gas_quantity],
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
