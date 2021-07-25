from .utils import get_molecular_diameter
import pandas as pd


class Environment:
    def __init__(
        self,
        concentration,
        temperature,
        pressure,
        max_distance,
        active_gas,
        passive_gas,
        container_volume,
    ):
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
        self.passive_gas_quantity = self.get_passive_gas_quantity()
        self.active_gas_quantity = self.passive_gas_quantity * concentration

    def get_passive_gas_quantity(self):
        R = 8.314  # universal gas constant
        NA = 6.0221409e23  # avogadro's number
        num = NA * self.pressure * self.container_volume
        den = R * self.temperature
        return num / den

    @property
    def info_dataframe(self):
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

