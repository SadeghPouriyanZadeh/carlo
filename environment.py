from .utils import get_molecular_diameter


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
