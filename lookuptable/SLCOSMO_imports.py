import jax.numpy as jnp, numpy as np, numpyro.distributions as dist, astropy.units as u
from MGE_jax import MGE
from jax import random
from numpyro.infer import MCMC, NUTS
from astropy.cosmology import Planck18 as cosmo
from astropy import constants as const
import jam_sph_proj, param_util
from functools import partial
from numpyro.infer import init_to_median, init_to_sample, NUTS
import arviz as az