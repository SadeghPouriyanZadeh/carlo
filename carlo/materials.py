from collections import namedtuple

Material = namedtuple("Material", ["name", "density", "molweight"])

h2 = Material("H2", 86, 2.01588e-3)  # solid state
tio2_rutile = Material("TiO2 Rutile", 4.23e3, 79.866e-3)
tio2_anatase = Material("TiO2 Anatase", 3.78e3, 79.866e-3)
n2 = Material("N2", 1026.5, 28.0134e-3)  # solid state
o2 = Material("O2", 687.5, 32e-3)  # solid state
ar = Material("Ar", 1616.0, 39.948e-3)  # solid state
air = Material(
    "Air",
    0.78 * n2.density + 0.21 * o2.density + 0.01 * ar.density,
    0.78 * n2.molweight + 0.21 * o2.molweight + 0.01 * ar.molweight,
)
