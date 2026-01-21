from jax.scipy import special, ndimage, signal
# from scipy import
import jax.numpy as jnp
from time import perf_counter as clock
import matplotlib.pyplot as plt
import jax
import nquad as nq

##############################################################################

def bilinear_interpolate(xv, yv, im, xout, yout):
    """
    The input array has size im[ny, nx] as in the output
    of im = f(meshgrid(xv, yv))
    xv and yv are vectors of size nx and ny respectively.

    """
    ny, nx = jnp.shape(im)
    # assert (nx, ny) == (xv.size, yv.size), "Input arrays dimensions do not match"

    xi = (nx-1.)/(xv[-1] - xv[0]) * (xout - xv[0])
    yi = (ny-1.)/(yv[-1] - yv[0]) * (yout - yv[0])
 
    a = ndimage.map_coordinates(im.T, jnp.array([xi, yi]), order=1, mode='nearest')

    return a

##############################################################################

def ibetam(a, b, x):
    """
    Incomplete beta function defined as the Mathematica Beta[x, a, b]:
    Beta[x, a, b] = Integral[t^(a - 1) * (1 - t)^(b - 1), {t, 0, x}]
    This routine only works for (0 < x < 1) & (b > 0) as required by JAM.

    """
    # V1.0: Michele Cappellari, Oxford, 01/APR/2008
    # V2.0: Use Hypergeometric function for negative a or b.
    #    Eq.(8.17.7) of Olver et al. (2010) https://dlmf.nist.gov/8.17.E7
    #    MC, Oxford, 04/APR/2008
    # V3.0: Use recurrence relation for (a < 0) & (b > 0)
    #    Eq.(8.17.20) of Olver et al. (2010) https://dlmf.nist.gov/8.17.E20
    #    After suggestion by Gary Mamon. MC, Oxford, 16/APR/2009

    a = a + 4.76123e-7  # Perturb to avoid singularities in gamma and betainc
    if jnp.all(a > 0):
        ib = special.betainc(a, b, x)
    else:
        p = jnp.ceil(abs(jnp.min(a)))
        j = jnp.arange(p) + a
        gam1 = special.gamma(j + b)
        gam2 = special.gamma(j + 1)
        tot = (gam1/gam2*x**j).sum(1)
        ib = tot[:, None]*(1 - x)**b/special.gamma(b) + special.betainc(a + p, b, x)

    return ib*special.beta(a, b)



# def ibetam(a, b, x):
#     """
#     Incomplete beta function defined as the Mathematica Beta[x, a, b]:
#     Beta[x, a, b] = Integral[t^(a - 1) * (1 - t)^(b - 1), {t, 0, x}]
#     This routine only works for (0 < x < 1) & (b > 0) as required by JAM.
#     """
#     a = a + 4.76123e-7

#     def true_branch(_):
#         return special.betainc(a, b, x)

#     def false_branch(_):
#         p = jnp.ceil(abs(jnp.min(a)))
#         j = jnp.arange(p) + a
#         gam1 = special.gamma(j + b)
#         gam2 = special.gamma(j + 1)
#         tot = (gam1/gam2*x**j).sum(1)
#         ib = tot[:, None]*(1 - x)**b/special.gamma(b) + special.betainc(a + p, b, x)
#         return ib

#     ib = jax.lax.cond(jnp.all(a > 0), true_branch, false_branch, operand=None)

#     return ib * special.beta(a, b)


##############################################################################

def integrand2d(s, t, sigma_lum, sigma_pot, dens_lum, mass, Mbh, rmin, beta, tensor):
    """
    This 2-dim integral is used when the Jeans LOS integral is not analytic

    """
    # TANH Change of variables for the LOS r-integral
    # jnp.log([rmin, rmax]) -> [R, inf]
    drds = jnp.exp(s)
    r = rmin + drds

    # TANH Change of variables for Jeans r1-integral (Sec.6.2 of Cappellari 2020, MNRAS, 494, 4819)
    # jnp.log([rmin, rmax]) -> [r, inf]
    dr1dt = jnp.exp(t)
    r1 = r + dr1dt

    ra, beta0, betainf, alpha = beta
    fun = (r1/r)**(2*beta0)
    fun *= ((1 + (r1/ra)**alpha)/(1 + (r/ra)**alpha))**(2*(betainf - beta0)/alpha)
    beta = beta0 + (betainf - beta0)/(1 + (ra/r)**alpha)

    G = 0.004301  # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    h = r1[:,:,None]/(jnp.sqrt(2)*sigma_pot[None,None,:])

    mass_r = Mbh + (mass*(special.erf(h) - 2/jnp.sqrt(jnp.pi)*h*jnp.exp(-h**2))).sum(-1) # eq.(49) of Cappellari (2008)
    chi = r1[:,:,None]/sigma_lum[None,None,:]
    nu_r = (dens_lum*jnp.exp(-0.5*(chi)**2)).sum(-1) # eq.(47) of Cappellari (2008)
    nuv2r_integ = G*nu_r*mass_r*fun/r1**2 # eq.(40) of Cappellari (2008)

    # LOS projection
    if tensor == 'los':
        qalpha = 1 - beta*(rmin/r)**2           # eq.(B8a) of Cappellari (2020)
    elif tensor == 'pmr':
        qalpha = 1 - beta + beta*(rmin/r)**2    # eq.(B8b) of Cappellari (2020)
    elif tensor == 'pmt':
        qalpha = 1 - beta                       # eq.(B8c) of Cappellari (2020)

    nuv2los_integ = 2*nuv2r_integ*qalpha*r/jnp.sqrt(r**2 - rmin**2)

    return nuv2los_integ*drds*dr1dt


##############################################################################

def integrand1d(t, sig_lum, sig_pot, dens_lum, mass, Mbh, rmin, beta, tensor, ra):
    """
    This function implements the integrand of equation (50) of Cappellari (2008, hereafter C08).
    Also implemented are the expressions for the proper motion components (pmr, pmt) from 
    Appendix B3 of Cappellari (2020, https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C)

    """ 
    # TANH Change of variables for the LOS r-integral
    # jnp.log([rmin, rmax]) -> [R, inf]
    drdt = jnp.exp(t)
    r = rmin + drdt[:, None]    # Broadcast over radii and MGE parameters

    if ra is not None:  # Osipkov-Merritt anisotropy: eq.A12 of Mamon & Lokas (2005, MNRAS)

        func = (2*ra**2 + rmin**2)/(ra**2 + rmin**2)**1.5 \
               * (r**2 + ra**2)/r**2 * jnp.arctan(jnp.sqrt((r**2 - rmin**2)/(ra**2 + rmin**2))) \
               - rmin**2/(ra**2 + rmin**2)*jnp.sqrt(r**2 - rmin**2)/r**2
    #jax error!
    # elif jnp.all(beta == 0):           # Simpler formula for isotropic case
    #     func = 2*jnp.sqrt(r**2 - rmin**2)/r**2

    else:
        #jax error!
        # if jnp.all(beta == beta[0]):
        #     beta = beta[0]          # faster calculation when beta is a scalar

        w = (rmin/r)**2
        bwp = ibetam(beta + 0.5, 0.5, w)
        bwm = (beta*bwp + jnp.sqrt(1 - w)*w**(beta - 0.5))/(beta - 0.5)  # = ibetam(beta - 0.5, 0.5, w) https://dlmf.nist.gov/8.17#E20
        gam = special.gamma(beta - 0.5)
        gap = gam*(beta - 0.5)              # = special.gamma(beta + 0.5) https://dlmf.nist.gov/5.5#E1
        pg = jnp.sqrt(jnp.pi)/special.gamma(beta)

        k = w**(1 - beta)/rmin
        a = pg*gam - bwm                    # eq.(B11a) of Cappellari (2020)
        b = pg*gap - beta*bwp               # eq.(B11b) of Cappellari (2020)

        if tensor == 'los':
            func = k*(a - b)                # eq.(B10a) of Cappellari (2020)
        elif tensor == 'pmr':
            func = k*((1 - beta)*a + b)     # eq.(B10b) of Cappellari (2020)
        elif tensor == 'pmt':
            func = k*(1 - beta)*a           # eq.(B10c) of Cappellari (2020)

    nuf = (func*dens_lum*jnp.exp(-0.5*(r/sig_lum)**2)).sum(1)     # eq.(B4) of Cappellari (2020)
    G = 0.004301  # (km/s)^2 pc/Msun [6.674e-11 SI units (CODATA2018)]
    h = r/(jnp.sqrt(2)*sig_pot)
    mass_r = Mbh + (mass*(special.erf(h) - 2/jnp.sqrt(jnp.pi)*h*jnp.exp(-h**2))).sum(1)    # eq.(B6) of Cappellari (2020)

    integ = G*nuf*mass_r   # Vector of values computed at different radii

    return integ*drdt

##############################################################################

def second_moment(R, sig_lum, sig_pot, dens_lum, mass, Mbh, beta, logistic,
                  tensor, ra, sigmaPsf, normPsf, nrad, surf_lum, pixSize,
                  epsrel, N=1000):
    """
    This routine gives the second V moment after convolution with a PSF.
    The convolution is done using interpolation of the model on a
    polar grid, as described in Appendix A of Cappellari (2008).

    """
    psfConvolution = (jnp.max(sigmaPsf) > 0) * (pixSize > 0)
    lim = jnp.log(jnp.array([1e-6*jnp.median(sig_lum), 3*jnp.max(sig_lum)]))

    # # if psfConvolution: # PSF convolution

    # # Kernel step is 1/4 of largest value between sigma(min) and 1/2 pixel side.
    # # Kernel half size is the sum of 3*sigma(max) and 1/2 pixel diagonal.
    # #
    # # if step == 0:
    # # step = jnp.where(step ==0, jnp.min(sigmaPsf)/4, step)
    # mx = 3*jnp.max(sigmaPsf) + pixSize/jnp.sqrt(2)

    # # Make grid linear in log of radius RR
    # #
    # rmax = jnp.max(R) + mx # Radius of circle containing all data + convolution
    # #step = rmax/N##################################################################
    # factor = 5
    # step = 484/factor
    # step = 50
    # rr = jnp.geomspace(step/jnp.sqrt(2), rmax, nrad)   # Linear grid in jnp.log(rr)
    # logRad = jnp.log(rr)

    # # The model Vrms computation is only performed on the radial grid
    # # which is then used to interpolate the values at any other location
    # #
    
    # def func(r_):
    #     args = (sig_lum, sig_pot, dens_lum, mass, Mbh,  r_, beta, tensor)
        
    #     if logistic:
    #         wm2res = nq.nquad(integrand2d, jnp.array([lim, lim]), args=args)
    #     else:
    #         wm2res = nq.quad(integrand1d, jnp.array(lim), args=args+(ra,))[0]
    #     return wm2res
    
    # vfunc = jax.vmap(func,0,0)
    # wm2Pol = vfunc(rr) # Integration of equation (50)

    # nx = 400 #100 #rmax / step #jnp.ceil(rmax/step).astype(int)
    # num_nx = 2*nx
    # #step = rmax/N ########################################################
    # x1 = jnp.linspace(0.5 - nx, nx - 0.5, num_nx)*step
    # xCar, yCar = jnp.meshgrid(x1, x1)  # Cartesian grid for convolution

    # # Interpolate MGE model and Vrms over cartesian grid.
    # # Division by mgePol before interpolation reduces interpolation error
    # r = jnp.sqrt(xCar**2 + yCar**2)
    # mgeCar = (surf_lum*jnp.exp(-0.5*(r[..., None]/sig_lum)**2)).sum(-1)
    # mgePol = (surf_lum*jnp.exp(-0.5*(rr[:, None]/sig_lum)**2)).sum(-1)
    # wm2Car = mgeCar*jnp.interp(jnp.log(r), logRad, wm2Pol/mgePol)

    # nk = 150 #100 #jnp.ceil(mx/step).astype(int)
    # num_nk = 2*nk+1
    # #step = mx // N + 1#####################################################################
    # kgrid = jnp.linspace(-nk, nk, num_nk)*step
    # xgrid, ygrid = jnp.meshgrid(kgrid, kgrid) # Kernel is square

    # # Compute kernel with equation (A6) of Cappellari (2008).
    # # Normalization is irrelevant here as it cancels out.
    # #
    # dx = pixSize/2
    # sp = jnp.sqrt(2)*sigmaPsf
    # xg, yg = xgrid[..., None], ygrid[..., None]
    # kernel = normPsf*(special.erf((dx - xg)/sp) + special.erf((dx + xg)/sp)) \
    #                 *(special.erf((dx - yg)/sp) + special.erf((dx + yg)/sp))
    # kernel = jnp.sum(kernel, 2)   # Sum over PSF components

    # # Seeing and aperture convolution with equation (A3) of Cappellari (2008)
    # #

    # m1, m2 = signal.fftconvolve(jnp.array([wm2Car, mgeCar]), kernel[None, ...], mode='same')
    # #m1 = jax.scipy.signal.convolve( wm2Car, kernel, mode='same',method='fft')
    # #m2 = jax.scipy.signal.convolve( mgeCar, kernel, mode='same',method='fft')


    # muCar = jnp.sqrt(m1/m2)

    # # Interpolate convolved image at observed apertures.
    # # Aperture integration was already included in the kernel.
    
    # sigp = bilinear_interpolate(x1, x1, muCar, R/jnp.sqrt(2), R/jnp.sqrt(2))

    
    #else: # No PSF convolution: just compute values

        # assert jnp.all(R > 0), "One must avoid the singularity at `R = 0`"

    def body_fn(carry, rj):
        # 这里可以把每个循环的计算独立成一个函数
        args = (sig_lum, sig_pot, dens_lum, mass, Mbh, rj, beta, tensor)
        if logistic:
            wm2Pol = nq.nquad(integrand2d, jnp.array([lim, lim]), args=args)
        else:
            wm2Pol = nq.quad(integrand1d, lim, args=args+[ra])
        mgePol = jnp.sum(surf_lum * jnp.exp(-0.5 * (rj / sig_lum) ** 2))
        sigp_val = jnp.sqrt(wm2Pol / mgePol)
        return carry, sigp_val

    _, sigp = jax.lax.scan(body_fn, None, R)
    psfConvolution = None

    return sigp, psfConvolution

##############################################################################

class jam_sph_proj:

    """
    PURPOSE
    -------

    This procedure calculates a prediction for any of the three components of
    the
    the projected second velocity moments V_RMS = sqrt(V^2 + sigma^2), or for a
    non-rotating galaxy V_RMS = sigma, for an anisotropic spherical galaxy
    model.
    It implements the solution of the anisotropic Jeans equations
    presented in equation (50) of `Cappellari (2008, MNRAS, 390, 71).
    <https://ui.adsabs.harvard.edu/abs/2008MNRAS.390...71C>`_
    PSF convolution is done as described in the Appendix of Cappellari (2008).
    This procedure includes the proper motions calculation given in
    Appendix B3 of `Cappellari (2020, MNRAS, 494, 4819)
    <https://ui.adsabs.harvard.edu/abs/2020MNRAS.494.4819C>`_

    CALLING SEQUENCE
    ----------------

    .. code-block:: python

        from jampy.jam_sph_proj import jam_sph_proj

        jam = jam_sph_proj(surf_lum, sigma_lum, surf_pot, sigma_pot, mbh,
            distance, rad, beta=None, data=None, epsrel=1e-2, errors=None,
            ml=None, normpsf=1, nrad=50, pixsize=0, plot=True, quiet=False,
            rani=None, sigmapsf=0, step=0, tensor='los')
        sigma_los_model = jam.model

    INPUT PARAMETERS
    ----------------

    surf_lum:
        vector of length N containing the peak surface brightness of the
        MGE Gaussians describing the galaxy surface brightness in units of
        Lsun/pc^2 (solar luminosities per parsec^2).
    sigma_lum:
        vector of length N containing the dispersion in arcseconds of
        the MGE Gaussians describing the galaxy surface brightness.
    surf_pot:
        vector of length M containing the peak value of the MGE Gaussians
        describing the galaxy surface density in units of Msun/pc^2 (solar
        masses per parsec^2). This is the MGE model from which the model
        potential is computed.

        In a common usage scenario, with a self-consistent model, one has
        the same Gaussians for both the surface brightness and the potential.
        This implies SURF_POT = SURF_LUM, SIGMA_POT = SIGMA_LUM.
        The M/L, by which SURF_POT has to be multiplied to best match the
        data, is fitted by the routine when passing the RMS and ERMS
        keywords with the observed kinematics.

    sigma_pot:
        vector of length M containing the dispersion in arcseconds of
        the MGE Gaussians describing the galaxy surface density.
    mbh:
        Mass of a nuclear supermassive black hole in solar masses.

        VERY IMPORTANT: The model predictions are computed assuming SURF_POT
        gives the total mass. In the common self-consistent case one has
        SURF_POT = SURF_LUM and if requested (keyword ML) the program can scale
        the output RMSMODEL to best fit the data. The scaling is equivalent to
        multiplying *both* SURF_POT and MBH by a factor M/L. To avoid mistakes,
        the actual MBH used by the output model is printed on the screen.
    distance:
        distance of the galaxy in Mpc.
    rad:
        Vector of length P with the (positive) radius from the galaxy center
        in arcseconds of the bins (or pixels) at which one wants to compute
        the model predictions.

        When no PSF/pixel convolution is performed (SIGMAPSF=0 or PIXSIZE=0)
        there is a singularity at RAD=0 which must be avoided.

    OPTIONAL KEYWORDS
    -----------------

    beta: array_like with shape (n,) or (4,)
        Radial anisotropy of the individual kinematic-tracer MGE Gaussians
        (Default: ``beta=jnp.zeros(n)``)::

            beta = 1 - (sigma_th/sigma_r)^2  # with align=`sph`

        When ``logistic=True`` the procedure assumes::

            beta = [r_a, beta_0, beta_inf, alpha]

        and the anisotropy of the whole JAM model varies as a logistic
        function of the logarithmic spherical radius::

            beta(r) = beta_0 + (beta_inf - beta_0)/[1 + (r_a/r)^alpha]

        Here ``beta_0`` represents the anisotropy at ``r = 0``, ``beta_inf``
        is the anisotropy at ``r = inf`` and ``r_a`` is the anisotropy
        transition radius, with ``alpha`` controlling the sharpness of the
        transition. In the special case ``beta_0 = 0, beta_inf = 1, alpha = 2``
        the anisotropy variation reduces to the form by Osipkov & Merritt, but
        the extra parameters allow for much more realistic anisotropy profiles.
    data:
        Vector of length P with the input observed stellar
        V_RMS=sqrt(velBin^2 + sigBin^2) at the coordinates positions
        given by the vector RAD.

        If RMS is set and ML is negative or not set, then the model is fitted to
        the data, otherwise the adopted ML is used and just the chi^2 is
        returned.
    epsrel: float, optional
        Relative error requested for the numerical quadrature
        (Default: ``epsrel=1e-2``)
    errors:
        Vector of length P with the 1sigma errors associated to the RMS
        measurements. From the error propagation
        ERMS = sqrt((dVel*velBin)^2 + (dSig*sigBin)^2)/RMS,
        where velBin and sigBin are the velocity and dispersion in each bin
        and dVel and dSig are the corresponding errors
        (Default: constant errors=0.05*MEDIAN(data)).
    logistic: bool, optional
        When ``logistic=True``, JAM interprets the anisotropy parameter
        ``beta`` as defining a 4-parameters logistic function.
        See the documentation of the anisotropy keyword for details.
        (Default ``logistic=False``)
    ml:
        Mass-to-light ratio to multiply the values given by SURF_POT.
        Setting this keyword is completely equivalent to multiplying the
        output RMSMODEL by SQRT(M/L) after the fit. This implies that the
        BH mass becomes MBH*(M/L).

        If this keyword is set to a negative number in input, the M/L is
        fitted from the data and the keyword returns the best-fitting M/L
        in output. The BH mass of the best-fitting model is MBH*(M/L).
    normpsf:
        Vector of length Q with the fraction of the total PSF flux
        contained in the various circular Gaussians describing the PSF of the
        observations. It has to be total(NORMPSF) = 1. The PSF will be used for
        seeing convolution of the model kinematics.
    nrad:
        Number of logarithmically spaced radial positions for which the
        models is evaluated before interpolation and PSF convolution. One may
        want to increase this value if the model has to be evaluated over many
        orders of magnitutes in radius (default: NRAD=50).
    pixsize:
        Size in arcseconds of the (square) spatial elements at which the
        kinematics is obtained. This may correspond to the size of the spaxel
        or lenslets of an integral-field spectrograph. This size is used to
        compute the kernel for the seeing and aperture convolution.

        If this is not set, or PIXSIZE = 0, then convolution is not performed.
    plot:
        Set this keyword to produce a plot at the end of the calculation.
    quiet:
        Set this keyword not to print values on the screen.
    rani: float, optional
        If this keyword is set to a numeric value, the program assumes the
        Osipkov-Merritt anisotropy with anisotropy radius ``rani``.
        Setting this keyword is equivalent to setting ``beta = [rani, 0, 1, 2]``
        but for this special anisotropy profile one of the two numerical
        quadratures is analytic and this was useful for testing.
    sigmapsf:
        Vector of length Q with the dispersion in arcseconds of the
        circular Gaussians describing the PSF of the observations.

        If this is not set, or SIGMAPSF = 0, then convolution is not performed.

        IMPORTANT: PSF convolution is done by creating a 2D image, with pixels
        size given by STEP=MAX(SIGMAPSF,PIXSIZE/2)/4, and convolving it with
        the PSF + aperture. If the input radii RAD are very large with respect
        to STEP, the 2D image may require a too large amount of memory. If this
        is the case one may compute the model predictions at small radii
        separately from those at large radii, where PSF convolution is not
        needed.
    step:
        Spatial step for the model calculation and PSF convolution in arcsec.
        This value is automatically computed by default as
        STEP=MAX(SIGMAPSF,PIXSIZE/2)/4. It is assumed that when PIXSIZE or
        SIGMAPSF are big, high resolution calculations are not needed. In some
        cases however, e.g. to accurately estimate the central Vrms in a very
        cuspy galaxy inside a large aperture, one may want to override the
        default value to force smaller spatial pixels using this keyword.

        Use this keyword to set the desired scale of the model when no PSF or
        pixel convolution is performed (SIGMAPSF=0 or PIXSIZE=0).
    tensor:
        Specifies any of the three component of the projected velocity
        dispersion
        tensor one wants to calculate. Possible options are (i) "los" (
        default) for
        the line-of-sight component; (ii) "pmr" for tha radial component of
        the proper
        motion second moment and (iii) "pmt" for the tangential component of
        the proper
        motion second moment. All three components are computed in km/s at
        the adopted
        distance.

    OUTPUT PARAMETERS
    -----------------

    attributes of the ``jam_sph_proj`` class

    .model: array_like with shape (p,)
        Model predictions for the velocity second moments (sigma in the
        spherical non-rotating case) of each bin. The line-of-sigh component
        or either of the proper motion components can be obtained in output
        using the ``tensor`` keyword.
    .ml:
        best fitting M/L.
    .chi2:
        Reduced chi^2 describing the quality of the fit::

            chi^2 = total( ((rms-rmsModel)/erms)^2 ) / n_elements(rms)

    .flux: array_like with shape (p,)
        PSF-convolved MGE surface brightness of each bin in ``Lsun/pc^2``,
        useful to plot the isophotes of the kinematic-tracer on the model
        results.
    """

    @staticmethod
    def get_kinematics(surf_lum, sigma_lum, surf_pot, sigma_pot, mbh, distance,
                 rad, beta=None, data=None, epsrel=1e-2, errors=None,
                 logistic=False, ml=None, normpsf=1, nrad=50, pixsize=0,
                 quiet=False, rani=None, sigmapsf=0, step=0,
                 tensor='los', N = 1000):

        if beta is None:
            beta = jnp.zeros_like(surf_lum)
        # assert (surf_lum.size == sigma_lum.size) and \
            #    ((len(beta) == 4 and logistic) or (len(beta) == surf_lum.size)), \
            # "The luminous MGE components and anisotropies do not match"
        # assert len(surf_pot) == len(sigma_pot), 'surf_pot and sigma_pot must have the same length'
        # assert tensor in ["los", "pmr", "pmt"], 'tensor must be: los, pmr or pmt'
        # if rani is not None:
            # assert tensor == 'los', "Only tensor='los' implemented for Osipkov-Merritt"
        if (errors is None) and (data is not None):
            errors = jnp.full_like(data, jnp.median(data)*0.05)  # Constant ~5% errors

        sigmapsf = jnp.atleast_1d(sigmapsf)
        normpsf = jnp.atleast_1d(normpsf)
        # assert sigmapsf.size == normpsf.size, "sigmaPSF and normPSF do not match"
        # assert round(jnp.sum(normpsf), 2) == 1, "PSF not normalized"

        pc = distance*jnp.pi/0.648 # Constant factor to convert arcsec --> pc

        sigmapsf_pc = sigmapsf*pc
        pixsize_pc = pixsize*pc
        step_pc = step*pc

        integ = 'quad1d'
        if len(beta) == 4 != surf_lum.size:  # Assumes beta = [r_a, beta_0, beta_inf, alpha]
            integ = 'quad2d'
            beta = beta.copy()
            beta = beta.at[0].set(beta[0]*pc)

        if rani is not None:
            rani = rani*pc

        sigma_lum_pc = sigma_lum*pc     # Convert from arcsec to pc
        dens_lum = surf_lum/(jnp.sqrt(2*jnp.pi)*sigma_lum_pc)

        sigma_pot_pc = sigma_pot*pc     # Convert from arcsec to pc
        mass = 2*jnp.pi*surf_pot*sigma_pot_pc**2

        t = clock()
        model, psfConvolution = second_moment(rad*pc, sigma_lum_pc,
            sigma_pot_pc, dens_lum, mass, mbh, beta, logistic, tensor, rani,
            sigmapsf_pc, normpsf, nrad, surf_lum, pixsize_pc, epsrel, N = N)

        if not quiet:
            print(f'jam_sph_proj elapsed time sec: {clock() - t:.2f} ({integ})')
            if not psfConvolution:
                txt = "No PSF convolution:"
                if jnp.max(sigmapsf) == 0:
                    txt += " sigmapsf == 0;"
                if pixsize == 0:
                    txt += " pixsize == 0;"
                print(txt)

        # Analytic convolution of the MGE model with an MGE circular PSF
        # using Equations (4,5) of Cappellari (2002, MNRAS, 333, 400).
        # Broadcast triple loop over (n_MGE, n_PSF, n_bins)
        sigma2 = sigma_lum**2 + sigmapsf[:, None]**2
        surf_conv = surf_lum*sigma_lum**2*normpsf[:, None]/jnp.sqrt(sigma2)
        flux = surf_conv[..., None]*jnp.exp(-0.5*rad**2/sigma2[..., None])
        flux = flux.sum((0, 1))  # PSF-convolved Lsun/pc^2

        ####### Output and optional M/L fit
        # If `data`` keyword is not given all this section is skipped

        chi2 = 0.0
        if data is not None:

            if (ml is None) or (ml <= 0):

                # eq. (51) of Cappellari (2008, MNRAS)
                d, m = data/errors, model/errors
                scale = (d @ m)/(m @ m)
                ml = scale**2

            else:
                scale = jnp.sqrt(ml)

            model *= scale
            chi2 = jnp.sum(((data - model)/errors)**2)/data.size

            if not quiet:
                print(f'beta[1]={beta[1]:.2f}; M/L={ml:#.3g}; '
                      f'BH={mbh*ml:#.3g}; chi2/DOF={chi2:#.3g}')
                print(f'Total mass MGE: {jnp.sum(mass*ml):#.4g}')
            

        return model, chi2, flux

##############################################################################

    @staticmethod
    def plot(rad,model):
        rad1 = rad #.clip(0.38*self.pixsize)
        plt.clf()
        # plt.errorbar(rad1, self.data, yerr=self.errors, fmt='o')
        plt.plot(rad1, model, 'r')
        plt.ylim(260,400)
        plt.xscale('log')
        plt.xlabel('R (arcsec)')
        plt.ylabel(r'$\sigma$ (km/s)')

##############################################################################

def test_jam_sph_proj():

    import numpy as np
    surf_pc = np.array([6229., 3089., 5406., 8443., 4283., 1927., 708.8, 268.1, 96.83])
    sigma_arcsec = np.array([0.0374, 0.286, 0.969, 2.30, 4.95, 8.96, 17.3, 36.9, 128.])

    # Realistic observed stellar kinematics. It comes from AO observations
    # at R < 2" and seeing-limited long slit observations at larger radii.
    # The galaxy has negligible rotation, and we can use sigma as V_RMS
    #
    sig = np.array([395., 390., 387., 385., 380., 365., 350., 315., 310., 290., 260.])  # km/s
    dsig = sig*0.02  # assume 2% errors in sigma
    rad = np.array([0.15, 0.2, 0.3, 0.4, 0.5, 1, 1.5, 3, 5, 9, 15])  # arcsec

    # Assume the anisotropy variation is described by the function
    # beta(r) = beta_0 + (beta_inf - beta_0)/[1 + (r_a/r)^alpha]
    r_a = 1         # Anisotropy transition radius in arcsec
    beta_0 = -0.1   # Inner anisotropy
    beta_inf = 0.5  # Outer anisotropy
    alpha = 1       # Sharpness of the anisotropy transition
    beta = jnp.array([r_a, beta_0, beta_inf, alpha])

    # Compute V_RMS profiles and optimize M/L to best fit the data.
    # Assume self-consistency: same MGE for luminosity and potential.
    #
    pixSize = 0.1           # Spaxel size in arcsec
    sigmapsf = jnp.array([0.1, 0.6])   # sigma of the PSF in arcsec from AO observations
    normpsf = jnp.array([0.7, 0.3])
    mbh = 2e8               # Black hole mass in solar masses before multiplication by M/L
    distance = 20.          # Mpc


    jam_obj = jam_sph_proj()

    from functools import partial
    jam_eval = partial(jam_obj.get_kinematics, sigma_lum=sigma_arcsec, surf_pot=surf_pc,  sigma_pot=sigma_arcsec, mbh=mbh,
                    distance=distance, rad=rad, beta=beta, sigmapsf=sigmapsf,
                    normpsf=normpsf, pixsize=pixSize, data=sig, errors=dsig,
                    tensor='los', logistic=True, quiet=True)

    model,chi2, flux = jam_eval(surf_pc)


    derivative_jam_eval = jax.jacfwd(jam_eval)
    der_jam = derivative_jam_eval(surf_pc)

    print(der_jam[0].shape,model.shape,surf_pc.shape)

    jam_obj.plot(rad,model)
    plt.show()
