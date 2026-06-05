"""
DEM (Differential Emission Measure) analysis — instrument-agnostic.

Solvers and generic temperature-response tools. The solvers take
``(intensities, errors, response, temperatures)`` and are not tied to any
instrument; instrument-specific response wiring lives in the instrument
package (e.g. ``egghouse.sdo.get_temperature_response`` for SDO/AIA, which
calls :func:`temperature_response_from_chianti` here).

Solvers (each carries its reference in its docstring):
    SITES        — Morgan & Pickering (2019), Sol. Phys. 294, 135
    NNLS         — Lawson & Hanson (1995); Hannah & Kontar (2012), A&A 539, A146
    regularized  — Hannah & Kontar (2012), A&A 539, A146
    sparse       — Cheung et al. (2015), ApJ 807, 143
    plowman      — Plowman et al. (2013), ApJ 771, 2
    mcmc         — Kashyap & Drake (1998), ApJ 503, 450
    spline       — Weber et al. (2004), IAU Symp. 223, 321
    gaussian     — Aschwanden et al. (2013), Sol. Phys. 283, 5

Modules:
    response: generic temperature-response tools (grid, CHIANTI fold, SSW loader)
    sites/nnls/regularized/sparse/plowman/mcmc/spline/gaussian: solvers
    utils: map processing + derived quantities (EM, mean T, loci)
"""

from .response import (
    get_default_temperatures,
    temperature_response_from_chianti,
    load_ssw_temperature_response,
    compute_response_derivative,
    get_response_weights,
    HAS_FIASCO,
)
from .sites import (
    dem_sites,
    dem_sites_pixel,
)
from .nnls import (
    dem_nnls,
    calibrate_reg_scale,
)
from .sparse import dem_sparse
from .regularized import dem_regularized
from .plowman import dem_plowman
from .mcmc import dem_mcmc
from .spline import dem_spline
from .gaussian import dem_gaussian
from .utils import (
    dem_map,
    compute_dem_errors,
    get_emission_measure,
    get_mean_temperature,
)

__all__ = [
    # Generic response tools
    "get_default_temperatures",
    "temperature_response_from_chianti",
    "load_ssw_temperature_response",
    "compute_response_derivative",
    "get_response_weights",
    "HAS_FIASCO",
    # SITES algorithm
    "dem_sites",
    "dem_sites_pixel",
    # NNLS algorithm
    "dem_nnls",
    "calibrate_reg_scale",
    # Additional solvers
    "dem_sparse",
    "dem_regularized",
    "dem_plowman",
    "dem_mcmc",
    "dem_spline",
    "dem_gaussian",
    # Utilities
    "dem_map",
    "compute_dem_errors",
    "get_emission_measure",
    "get_mean_temperature",
]
