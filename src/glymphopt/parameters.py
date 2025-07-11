import numbers
import ufl
from typing import Any

import pint

ureg = pint.get_application_registry()
dimless = pint.Unit("")
percent = 0.01 * dimless
mm = ureg.mm
cm = ureg.cm
s = ureg.second
minute = ureg.minute
liters = ureg.liters


DEFAULT_PARAMETERS = {
    "n": {  # Volume fractions
        "e": 20 * percent,
        "p": 2 * percent,
        "b": 4 * percent,
        "sas": 80 * percent,
    },
    "D": {  # Diffusion coefficients
        "e": 1.3e-4 * mm**2 / s,
    },
    "gamma": (3.0 * dimless),  # Factor of enhanced diffusion in PVS
    "rho": 0.113 * dimless,  # Ratio of free diffusion of gadobutrol to water
    "r": {"b": 1e-3 * 1 / s},  # Decay rate
    "t": {  # Transfer coefficient (permeability * area-volume ratio)
        "ep": 2.9e-2 * 1 / s,
        "pb": 2.0e-6 * 1 / s,
    },
    "k": {  # Surface membrane conductivity
        "e": 1e-5 * mm / s,
        "p": 3.7e-4 * mm / s,
    },
    "beta": (1.71 * dimless),  # Ratio between blood volume and brain volume ()
}
DEFAULT_PARAMETERS = {
    "n": {  # Volume fractions
        "e": 20 * percent,
        "p": 2 * percent,
        "b": 4 * percent,
        "sas": 80 * percent,
    },
    "D": {  # Diffusion coefficients
        "e": 1.3e-4 * mm**2 / s,
    },
    "gamma": (3.0 * dimless),  # Factor of enhanced diffusion in PVS
    "rho": 0.113 * dimless,  # Ratio of free diffusion of gadobutrol to water
    "r": {"b": 1e-3 * 1 / s},  # Decay rate
    "t": {  # Transfer coefficient (permeability * area-volume ratio)
        "ep": 2.9e-2 * 1 / s,
        "pb": 2.0e-6 * 1 / s,
    },
    "k": {  # Surface membrane conductivity
        "e": 1e-5 * mm / s,
        "p": 3.7e-4 * mm / s,
    },
    "beta": (1.71 * dimless),  # Ratio between blood volume and brain volume ()
}
DEFAULT_UNITS = {
    "n": "",
    "D": "mm**2 / s",
    "t": "1 / s",
    "r": "1 / s",
    "beta": "",
    "k": "mm / s",
}


def get_default_parameters():
    default_parameters = {**DEFAULT_PARAMETERS}
    gamma = default_parameters["gamma"]
    De = default_parameters["D"]["e"]
    default_parameters["D"]["p"] = gamma * De
    return {**DEFAULT_PARAMETERS}


def unpack(p, *args):
    return tuple(p[i] for i in args)


def consume(p, *args):
    return tuple(p.pop(i) for i in args)


def get_dimless_parameters(T: pint.Quantity, X: pint.Quantity):
    default = get_default_parameters()
    n, D, pi, r, k, beta, gamma, rho = unpack(
        default, "n", "D", "t", "r", "k", "beta", "gamma", "rho"
    )
    return {
        "n": {i: (ni).to("") for i, ni in n.items()},
        "D": {i: (T / X**2 * Di).to("") for i, Di in D.items()},
        "t": {ij: (T * pi_ij).to("") for ij, pi_ij in pi.items()},
        "r": {i: (T * ri).to("") for i, ri in r.items()},
        "k": {i: (T / X * ki).to("") for i, ki in k.items()},
        "beta": beta,
        "gamma": gamma,
        "rho": rho,
    }


def singlecomp_parameters(threecomp_parameters=None):
    threecomp_parameters = threecomp_parameters or get_default_parameters()
    params = flatten_dict(threecomp_parameters)
    (
        n_e,
        n_p,
        n_sas,
        k_e,
        k_p,
        t_pb,
        gamma,
        rho,
    ) = consume(params, "n_e", "n_p", "n_sas", "k_e", "k_p", "t_pb", "gamma", "rho")
    phi = n_e + n_p
    return {
        "a": rho * (n_e + gamma * n_p) / phi,
        "r": t_pb / phi,
        "k": (k_e + k_p) / phi,
        "phi": phi,
        "phi_sas": n_sas,
    }


def flatten_dict(nested_dict, prefix=""):
    flat_dict = {}
    for key, val in nested_dict.items():
        if isinstance(val, dict):
            flat_dict = flat_dict | flatten_dict(val, f"{key}_")
        else:
            flat_dict[f"{prefix}{key}"] = val
    return flat_dict


def remove_units(params):
    """Converts all quantities to a dimless number."""
    dimless = {}
    for key, val in params.items():
        if isinstance(val, dict):
            dimless[key] = remove_units(val)
        else:
            dimless[key] = val.magnitude
    return dimless


def print_quantities(p, offset, depth=0):
    """Pretty printing of dictionaries filled with pint.Quantities (or numbers)"""
    format_size = offset - depth * 2
    for key, value in p.items():
        if isinstance(value, dict):
            print(f"{depth * '  '}{str(key)}")
            print_quantities(value, offset, depth=depth + 1)
        else:
            if is_quantity(value):
                print(f"{depth * '  '}{str(key):<{format_size + 1}}: {value:.3e}")
            else:
                print(f"{depth * '  '}{str(key):<{format_size + 1}}: {value}")


def is_quantity(x: Any) -> bool:
    return isinstance(x, pint.Quantity) or isinstance(x, numbers.Complex)


def parameters_2d_default() -> dict[str, float | ufl.Coefficient]:
    return {
        "a": 2.0,
        "phi": 0.22,
        "D_": 0.0002,
        "r": 2e-6,
        "k": 0.02,
        "rho": 0.123,
        "eta": 0.4,
    }


if __name__ == "__main__":
    default = get_default_parameters()
    print_quantities(default, offset=4)
    print_quantities(singlecomp_parameters(default), offset=4)
