from collections import namedtuple

Material = namedtuple("Material", ["density", "molweight"])

h2 = Material(86, 2.01588e-3)  # solid state
tio2_rutile = Material(4.23e3, 79.866e-3)
tio2_anatase = Material(3.78e3, 79.866e-3)
n2 = Material(1026.5, 28.0134e-3)  # solid state
o2 = Material(687.5, 32e-3)  # solid state
ar = Material(1616.0, 39.948e-3)  # solid state
air = Material(
    0.78 * n2.density + 0.21 * o2.density + 0.01 * ar.density,
    0.78 * n2.molweight + 0.21 * o2.molweight + 0.01 * ar.molweight,
)
