'''
A collection of utilities shared between scripts.
'''
import numpy as np

import healpy as hp
from pixell import enmap, curvedsky
from optweight import (map_utils, mat_utils, alm_utils, alm_c_utils, sht,
                       solvers, preconditioners, type_utils)

def slice2len(sel):
    '''
    Convert slice object to number of resulting items.

    Parameters
    ----------
    sel : slice
       Slice.

    Returns
    -------
    length : int
        Number of items resulting from slice.
    '''

    return np.arange(sel.start, sel.stop, sel.step).size

def get_radii_leo(oversample=1):
    '''
    Get radii (in Mpc) suitable for the Local, Equilateral and Orthogonal
    bispectra.

    Parameters
    ----------
    oversample : float
        Over- or undersample the radii by this factor.

    Returns
    -------
    radii : (nr) array
        Radii in Mpc.
    '''

    low1 = np.linspace(1, 9377, num=int(24 * oversample), endpoint=False)
    rei1 = np.linspace(9377, 10007, num=int(4 * oversample), endpoint=False)
    rei2 = np.linspace(10007, 12632, num=int(8 * oversample), endpoint=False)
    rec1 = np.linspace(12632, 13682, num=int(12 * oversample), endpoint=False)
    rec2 = np.linspace(13682, 15500, num=int(75 * oversample), endpoint=False)
    rec3 = np.linspace(15500, 18000, num=int(5 * oversample), endpoint=False)

    radii = np.concatenate((low1, rei1, rei2, rec1, rec2, rec3))

    return radii

def get_radii_rec(oversample=1):
    '''
    Get truncated radii (in Mpc) centered on recombination.

    Parameters
    ----------
    oversample : float
        Over- or undersample the radii by this factor.

    Returns
    -------
    radii : (nr) array
        Radii in Mpc.
    '''

    radii = np.linspace(13700, 14100, num=int(40 * oversample), endpoint=False)

    return radii

def process_mask(mask, iquslice):
    '''
    Convert input mask to boolean mask.

    Parameters
    ----------
    mask : (npol, ny, nx) or (ny, nx) enmap
        Input mask.
    iquslice : slice
        Slice into IQU axis.

    Returns
    -------
    mask_bool : (npol, ny, nx)
        Boolean mask, True for good pixels. 

    Raises
    ------
    ValueError
        If input map does not match with input npol.

    Notes
    -----
    Npol of output mask can be, depending on iquslice, 1 (T),
    2 (Q, U) or 3 (I, Q, U).    
    '''

    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = mask.astype(bool)

    mask = mat_utils.atleast_nd(mask, 3)
    npol = mask.shape[0]

    if npol not in (1, 3):
        raise ValueError(f'{mask.shape=} not supported, npol != 1, 3')
    
    if npol == 3:
        mask = np.ascontiguousarray(mask[iquslice])
    elif npol == 1:        
        mask = mask * np.ones(slice2len(iquslice))[:,np.newaxis,np.newaxis]
        
    return mask

def compute_w3(mask):
    '''
    Return an approximate fsky factor.

    Parameters
    ----------
    mask : (npol, ny, nx) enmap
        Input mask.

    Returns
    -------
    w3 : (npol) array
        W3 factor per input mask.
    '''

    mask = mat_utils.atleast_nd(mask, 3)

    pmap = enmap.pixsizemap(mask.shape[-2:], mask.wcs)
    w3 = np.sum(((mask) ** 3) * pmap, axis=(-1, -2))
    w3 /= (4 * np.pi)

    return w3

def find_minfo(shape, wcs):
    '''
    Find map_info for input map.

    Parameters
    ----------
    shape : tuple
        Ny and Nx dimensions.
    wcs : astropy.wcs.WCS object
        Metainfo of map geometry.

    Returns
    -------
    minfo : optweight.map_utils.MapInfo object
        Metainfo mask
    '''

    for mtype in ['CC', 'fejer1',]:
        try:
            minfo = map_utils.match_enmap_minfo(
                shape, wcs, mtype=mtype)
        except ValueError:
            continue
        else:
            break

    return minfo

def get_b_ell(fwhm, lmax, iquslice, dtype=np.float64):
    """
    Get beam window function given FWHM.

    Parameters
    ----------
    fwhm : float
        FWHM in arcmin.
    lmax : int
        Max multipole.
    iquslice : slice
        Slice into IQU axis.
    dtype : type, optional
        Type of output.

    Returns
    -------
    b_ell : (npol, lmax + 1) array
        Gaussian beam window function.
    """

    npol = slice2len(iquslice)
    b_ell = np.ones((npol, lmax+1), dtype=dtype)
    b_ell[:] *= hp.gauss_beam(np.radians(fwhm / 60), lmax=lmax)

    return b_ell

def process_signal_spectra(spectra, lmax, no_te=False, dtype=np.float64):
    '''
    Return signal spectrum.

    Parameters
    ----------
    spectra : (4, nell) array
        TT, EE, BB and TE spectra units of D_ell.
    lmax : int
        Truncate to this multipole.
    no_te : bool, optional
        Set TE correlation to zero.
    dtype : type, optional
        Dtype of output.

    Returns
    -------
    cov_ell : (npol, npol, lmax + 1) array
        Output signal C_ells.

    Raises
    ------
    ValueError
        If input has wrong shape.
        If lmax is too high.
    '''

    if spectra.ndim != 2 or spectra.shape[0] != 4:
        raise ValueError(f'Wrong input shape: {spectra=}')

    ells = np.arange(lmax + 1)
    dells = ells * (ells + 1) / 2 / np.pi

    s_ell = np.zeros((3, 3, lmax + 1))

    if no_te:
        spectra[3] = 0

    s_ell[0,0,2:] = spectra[0,:lmax-1]
    s_ell[0,1,2:] = spectra[3,:lmax-1]
    s_ell[1,0,2:] = spectra[3,:lmax-1]
    s_ell[1,1,2:] = spectra[1,:lmax-1]
    s_ell[2,2,2:] = spectra[2,:lmax-1]

    s_ell[...,1:] /= dells[1:]

    return s_ell.astype(dtype, copy=False)

def slice_spectrum(cov_ell, pslice, lmax=None, lmin=None):
    '''
    Slice spectrum to desired polarizations and lmax.

    Parameters
    ----------
    cov_ell : (npol_in, npol_in, nell) array
        Input spectrum.
    pslice : slice, optional
        Slice into T, E, B axis.
    lmax : int, optional
        Maximum ell.
    lmin : int, optional
        Set values below lmin to zero.

    Returns
    -------
    cov_ell_out : (npol, npol, lmax + 1) array
        Sliced copy of input spectrum.

    Raises
    ------
    ValueError
        If lmax is too large.
        If input npol != 3.
    '''

    if lmax is None:
        lmax = cov_ell.shape[-1] - 1

    if lmax > (cov_ell.shape[-1] - 1):
        raise ValueError(
            f'{lmax=} exceeds {cov_ell.shape[-1]-1=} of input array.')
    
    if cov_ell.shape[0] != 3:
        raise ValueError(f'{cov_ell.shape=} does not have npol=3.')
    
    out = np.ascontiguousarray(cov_ell[pslice,pslice,:lmax+1])
    if lmin is not None:
        out[...,:lmin] = 0

    return out

def process_cov_wav(cov_wav, nl2d, iquslice, dtype=np.float64):
    '''
    Process wavelet covariance.

    Parameters
    ----------
    cov_wav : wavtrans.Wav object
        Input covariance matrix.
    nl2d : (npol, npol, nly, nlx) array
        Noise power specrum.
    iquslice : slice, optional
        Slice into I, Q, U axis.
    dtype : type, optional
        Dtype of output.

    Returns
    -------
    cov_wav : wavtrans.Wav object
        Sliced and recasted wavelet covariance.
    nl2d : (npol, npol, nly, nlx) array
        Sliced and recasted noise power specrum.
    '''

    if len(cov_wav.preshape) == 4:
        # Remove excess dimensions.
        for key in cov_wav.maps:
            cov_wav.maps[key] = cov_wav.maps[key][0,:,0,:]
        cov_wav.preshape = cov_wav.maps[key].shape[:-1]

    # If E-only or T-only, constrain wav_maps to only contain correct dims.
    for key in cov_wav.maps:
        cov_wav.maps[key] = cov_wav.maps[key][iquslice,iquslice]
        # Always make copy to ensure contiguous array.
        cov_wav.maps[key] = cov_wav.maps[key].astype(dtype, copy=True)
    cov_wav.preshape = cov_wav.maps[0,0].shape[:-1]
    cov_wav.dtype = dtype

    nl2d = nl2d[iquslice,iquslice,:,:]
    nl2d = nl2d.astype(dtype, copy=True)

    return cov_wav, nl2d

def process_icov_pix(icov_pix, iquslice, dtype=np.float64):
    '''
    If needed, slice into icov matrix. Ensure dtype.

    Parameters
    ----------
    icov_pix : (npol_in, npol_in, ny, nx) or (npol, ny, nx) enmap
        Per-pixel inverse covariance matrix.
    iquslice : slice, optional
        Slice into I, Q, U axis.

    Returns
    -------
    icov_pix : (npol_in, npol_in, ny, nx) or (npol, ny, nx) enmap
        Sliced and recasted inverse covariance matrix.

    Raises
    ------
    ValueError
        If input is not IQU.
    '''

    icov_pix = mat_utils.atleast_nd(icov_pix, 3)

    if icov_pix.shape[0] != 3:
        raise ValueError(f'{icov_pix.shape=}, npol is not 3.')
    
    if icov_pix.ndim == 3:
        icov_pix = icov_pix[iquslice].astype(dtype, copy=True)
    elif icov_pix.ndim == 4:
        icov_pix = icov_pix[iquslice,iquslice].astype(dtype, copy=True)
    else:
        raise ValueError(f'{icov_pix.shape=} not supported')

    return icov_pix

def init_solver(ainfo, minfo, icov_ell, b_ell, mask,
                spin, icov_pix=None, cov_wav=None, fkernels=None,
                cov_noise_2d=None, itau_ell=None, swap_bm=False,
                scale_a=False, lensop=None, no_masked_prec=False):
    '''
    Initialize CG solver and preconditioners.

    Parameters
    ----------
    ainfo : pixell.curvedsky.alm_info object
        Metainfo for output alms.
    minfo : map_utils.MapInfo object
        Metainfo for input map.
    icov_ell : (npol, npol, nell) or (npol, nell) array
        Inverse signal covariance. If diagonal, only the diagonal suffices.
    icov_pix : (npol, npol, npix) or (npol, npix) array
        Inverse noise covariance. If diagonal, only the diagonal suffices.
    b_ell : (npol, nell) array
        Beam window functions.
    mask = (npol, npix) array
        Pixel mask.
    spin : int, array-like
        Spin values for transform, should be compatible with npol.
    cov_wav : wavtrans.Wav object
        Wavelet block matrix representing the noise covariance.
    fkernels : fkernel.FKernelSet object
        Wavelet kernels.
    cov_noise_2d : (npol, npol, nly, nlx) array or (npol, nly, nlx) array, optional
        Noise covariance in 2D Fourier domain. If diagonal, only the
        diagonal suffices.
    itau_ell : (npol, npol, nell) array, optional
        Isotropic noise (co)variance.
    swap_bm : bool, optional
        If set, swap the order of the beam and mask operations. Helps convergence
        with large beams and high SNR data.
    scale_a : bool, optional
        If set, scale the A matrix to localization of N^-1 term. This may
        help convergence with small beams and high SNR data.
    lensop : lensing.LensAlm object
        Lensing instance used to compute lensing and adjoint lensing.
    no_masked_prec : float, optional
        If True, do not use the two masked preconditioners. Used for
        full sky data.

    Returns
    -------
    solver : optweight.solvers.CGWienerMap object
        Uninitialized solver object.
    prec_base : optweight.preconditioners object
        Either harmonic or pseudo inverse preconditioner.
    prec_masked_cg : optweight.preconditioners.MaskedPreconditionerCG object
        Preconditioner for masked pixels.
    prec_masked_mg : optweight.preconditioners.MaskedPreconditioner object
        Preconditioner for masked pixels.
    '''

    if scale_a:
        sfilt = mat_utils.matpow(b_ell, -0.5)
        lmax_mg = 3000
    else:
        sfilt = None
        lmax_mg = 6000

    npol = icov_ell.shape[0]
    imap_template = np.zeros((npol, minfo.npix), dtype=icov_ell.dtype)
    
    if icov_pix is not None:
        solver = solvers.CGWienerMap.from_arrays(
            imap_template, minfo, ainfo, icov_ell, icov_pix,
            b_ell=b_ell, mask_pix=mask, minfo_mask=minfo,
            draw_constr=False, spin=spin, swap_bm=swap_bm, sfilt=sfilt,
            lensop=lensop)

        prec_base = preconditioners.PseudoInvPreconditioner(
            ainfo, icov_ell, icov_pix, minfo, spin, b_ell=b_ell, sfilt=sfilt)

    elif icov_wav is not None:
        solver = solvers.CGWienerMap.from_arrays_fwav(
            imap_template, minfo, ainfo, icov_ell, cov_wav, fkernels,
            b_ell=b_ell, mask_pix=mask, minfo_mask=minfo,
            draw_constr=False, spin=spin, swap_bm=swap_bm, sfilt=sfilt,
            cov_noise_2d=cov_noise_2d, lensop=lensop)

        prec_base = preconditioners.HarmonicPreconditioner(
            ainfo, icov_ell, b_ell=b_ell, itau=itau_ell, sfilt=sfilt)

    if not no_masked_prec:
        prec_masked_cg = preconditioners.MaskedPreconditionerCG(
            ainfo, icov_ell, 0, mask.astype(bool), minfo, lmax=None,
            nsteps=15, lmax_r_ell=None, sfilt=sfilt)

        prec_masked_mg = preconditioners.MaskedPreconditioner(
            ainfo, icov_ell[0:1,0:1], 0, mask[0].astype(bool), minfo,
            min_pix=1000, n_jacobi=1, lmax_r_ell=lmax_mg, sfilt=sfilt)
    else:
        prec_masked_cg, prec_masked_mg = None, None

    return solver, prec_base, prec_masked_cg, prec_masked_mg

def compute_icov(imap, solver=None, prec_base=None, prec_masked_cg=None,
                 prec_masked_mg=None, niter_cg=0,  niter_mg=0,
                 two_level_cg=None, two_level_mg=None, no_masked_prec=False,
                 ofile=None, slice_output=True, verbose=False):
    '''
    Inverse-covariance filter input map using the conjugate gradient method.

    Parameters
    ----------
    imap : (npol, npix) array
        Input map.
    solver : optweight.solvers.CGWienerMap instance
        CG solver object.
    prec_base : callable
        The base preconditioner.
    prec_masked_cg : callable, optional
        Masked conjugate-gradient-based preconditioner.
    prec_masked_mg : callable. optional
        Masked multigrid-gradient-based preconditioner.    
    niter_cg : int
        Number of CG steps with prec_masked_cg.
    niter_mg : int
        Number of CG steps with prec_masked_mg.    
    two_level_cg : str, optional
        Type of 2-level precontioning using with CG masked preconditioner,
        pick from "ADEF-1", "ADEF-2".    
    two_level_mg : str, optional
        Type of 2-level precontioning using with MG masked preconditioner,
        pick from "ADEF-1", "ADEF-2".        
    no_masked_prec : float, optional
        If True, do not use the two masked preconditioners. Used for
        full sky data.
    ofile : str, optional
        Path to filename. If provided, write Wiener-filtered map to disk
        using this filename, e.g. "/path/to/sim.fits".
    slice_output : bool
        Slice the output alms based on input npol (i.e. discard B).
    verbose : bool, optional
        If set, print basic CG convergence metric.
    
    Returns
    -------
    icov_alm : (npol, nelem) complex array
        Spherical harmonic coefficients of the tnverse-covariance filtered
        input map.
    '''

    if (two_level_cg or two_level_mg) and no_masked_prec:
        raise ValueError(f'Cannot have both two_level_prec and no_masked_prec')

    solver.reset_preconditioner()
    solver.set_b_vec(imap)

    if two_level_cg:
        solver.add_preconditioner(
            preconditioners.get_2level_prec(
                prec_base, prec_masked_cg, solver, two_level_cg))
    elif two_level_cg is None:
        solver.add_preconditioner(prec_base)
        if not no_masked_prec:
            solver.add_preconditioner(prec_masked_cg)

    solver.init_solver()

    niter = niter_cg + niter_mg
    for idx in range(niter_cg):
        solver.step()
        print(solver.i, solver.err)

    solver.reset_preconditioner()
    solver.add_preconditioner(prec_base)

    if two_level_mg:
        solver.add_preconditioner(
            preconditioners.get_2level_prec(
                prec_base, prec_masked_mg, solver, two_level_mg,
                sel_masked=np.s_[0]))
    elif two_level_mg is None:
        if not no_masked_prec:
            solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])

    solver.b_vec = solver.b0
    if niter_cg == 0:
        solver.init_solver(x0=None)
    else:
        solver.init_solver(x0=solver.x)

    for idx in range(niter_cg, niter):

        solver.step()
        if verbose:
            print(solver.i, solver.err, f'rank : NOTIMPLEMENTED')

    if ofile is not None:
        hp.write_alm(ofile, solver.x, overwrite=True)        
    
    icov_out = solver.get_icov()
    npol = icov_out.shape[0]

    if slice_output:
        if npol == 2:
            # We have E, B. Only want E.
            oslice = slice(0, 1)
        elif npol == 3:
            # We have I, E, B. Only want T, E.
            oslice = slice(0, 2)

        elif npol == 1:
            oslice = np.s_[:]
    else:
        oslice = np.s_[:]
            
    return icov_out[oslice]

def draw_noise_wav(minfo, seed, dtype=np.float64, sqrt_cov_wav_op=None,
                   sqrt_n_op=None, fkernels=None):
    '''
    Draw a noise realization from a wavelet-based noise model.

    Parameters
    ----------
    minfo : optweight.map_utils.MapInfo object
        metainfo ouput map geometry.
    seed : int or np.random._generator.Generator object, optional
        Seed for np.random.seed.   
    dtype : type, optional
        Type for output map.
    sqrt_cov_wav_op : callable
        Function that applies square root wavelet of wavelet noise model.
    sqrt_n_op : callable
        Function that applied the square root of noise power spectrum.
    fkernels : fkernel.FKernelSet object
        Wavelet kernels.
    
    Returns
    -------
    omap : (npol, npix) array
        Map with noise realization.
    '''

    rng = np.random.default_rng(seed)

    sqrt_cov_wav = sqrt_cov_wav_op.m_wav
    wav_uni = noise_utils.unit_var_wav(
        sqrt_cov_wav.get_minfos_diag(), sqrt_cov_wav.preshape[:1],
        sqrt_cov_wav.dtype, seed=rng)

    rand_wav = sqrt_cov_wav_op(wav_uni)
    oshape = sqrt_cov_wav.preshape[-1:] + fkernels.shape_full
    fdraw = np.zeros(oshape, type_utils.to_complex(dtype))
    fdraw = wavtrans.wav2f(rand_wav, fdraw, fkernels)
    fdraw = sqrt_n_op(fdraw)

    omap = np.zeros((fdraw.shape[0], minfo.npix), dtype)
    dft.irfft(fdraw, map_utils.view_2d(omap, minfo))

    return omap

def draw_noise_pix(sqrt_cov_pix_op, minfo, seed, dtype):
    '''
    Draw a noise realization from a per-pixel noise model.

    Parameters
    ----------
    sqrt_cov_pix_op : callable
        Function that applies the square root of a pixel-based noise model.
    minfo : optweight.map_utils.MapInfo object
        metainfo ouput map geometry.
    seed : int or np.random._generator.Generator object, optional
        Seed for np.random.seed.   
    dtype : type, optional
        Type for output map.

    Returns
    -------
    omap : (npol, npix) array
        Map with noise realization.
    '''

    rng = np.random.default_rng(seed)

    npol = sqrt_cov_pix_op.m_pix.shape[0]
    omap = rng.standard_normal(
        npol * minfo.npix).reshape((npol, minfo.npix))
    omap = sqrt_cov_pix_op(omap)

    return omap

def draw_signal_alm(sqrt_cov_ell_op, ainfo, seed, dtype):
    '''
    Draw a signal realization.
    
    Parameters
    ----------
    sqrt_cov_ell_op : callable
        Function that applies the square root of the signal power spectrum.
    ainfo : pixell.curvedsky.alm_info object
        metainfo ouptut alm geometry.
    seed : int or np.random._generator.Generator object, optional
        Seed for np.random.seed.   
    dtype : type, optional
        Type for output alm.

    Returns
    -------
    alm : (npol, npix) array
        Spherical harmonic coefficients with signal realization.
    '''

    rng = np.random.default_rng(seed)

    npol = sqrt_cov_ell_op.m_ell.shape[0]
    unit_var_alm = alm_utils.unit_var_alm(
        ainfo, (npol,), rng)
    alm = sqrt_cov_ell_op(unit_var_alm)

    return alm.astype(dtype)

def alm_loader_template(seed, sqrt_cov_ell_op, b_ell,  minfo, ainfo, spin,
                        mask, dtype, sqrt_cov_pix_op=None,
                        wav_noise_opts=None, icov_opts=None):
    '''
    Generate signal + noise simulation and return the inverse-covariance
    filtered version.

    Parameters
    ----------
    seed : int or np.random._generator.Generator object, optional
        Seed for np.random.seed.   
    sqrt_cov_ell_op : callable
        Function that applies the square root of the signal power spectrum.    
    b_ell : (npol, nell)
        Beam window function.
    minfo : optweight.map_utils.MapInfo object
        metainfo ouput map geometry.
    ainfo : pixell.curvedsky.alm_info object
        metainfo internal alm geometry.
    spin : int or array-like
        Spin values for spherical harmonic transforms, should match npol.
    mask : (npol, npix) array
        Pixel mask.    
    dtype : type, optional
        Type for output map.
    sqrt_cov_pix_op : callable
        Function that applies the square root of a pixel-based noise model.
    wav_noise_opts : dict, optional
        Keyword arguments to `draw_noise_wav`.
    icov_opts : dict, optional
        Keyword arguments to `compute_icov`.

    Returns
    -------
    icov_alm : (npol, nelem) complex array
        Spherical harmonic coefficients of the tnverse-covariance filtered
        signal + noise realization.
    '''

    rng = np.random.default_rng(seed)

    alm = draw_signal_alm(
        sqrt_cov_ell_op, ainfo, rng, type_utils.to_complex(dtype))

    # Npol of alm should be either 1 (=T), 2 (=E, B) or 3 (=T, E, B).
    alm = icov_opts['solver'].lens(alm)

    alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
    omap = np.zeros((alm.shape[0], minfo.npix), dtype=dtype)
    sht.alm2map(alm, omap, ainfo, minfo, spin)

    # Only mask signal.
    omap *= mask

    if wav_noise_opts:
        omap += draw_noise_wav(minfo, rng, dtype=dtype,
                               **wav_noise_opts)
    else:
        omap += draw_noise_pix(sqrt_cov_pix_op, minfo, rng, dtype)

    return compute_icov(omap, **icov_opts)

def compute_icov_alm(alm, pslice, icov_opts):
    '''
    Inverse-covariance filter a set of alm coefficients by
    first applying the P projection matrix (d_pix = P slm + n).

    Parameters
    ----------
    alm : (npol, nelem) complex array
        Input alms.
    pslice : slice
        Slice into I, E, B that describes how input sits in
        an I, E, B vector.
    icov_opts : dict
        Keyword arguments for `compute_icov`.

    Returns
    -------
    oalm : (npol, nelem)
        Inverse-covariance filtered alms.

    Notes
    -----
    In case this function is called with just E-mode vector,
    pslice should be slice(1, 3), in which case an extra zero vector
    will be concatenated for B. In case input is T, E, pslice should
    be (0, 3). Again a zero vector will be concatenated. If input is
    just T, pslice should be (0, 1) and no extra vector is needed.
    '''

    npol = alm.shape[0]
    
    # Append zero B if needed.
    npol_ext = slice2len(pslice)

    if npol < npol_ext:
        alm_ext = np.zeros((npol_ext, alm.shape[-1]), dtype=alm.dtype)
        alm_ext[:npol] = alm
    else:
        alm_ext = alm
    
    omap = icov_opts['solver'].proj(alm_ext)

    return compute_icov(omap, **icov_opts)

def get_itotcov_ell(icov_signal_ell, icov_noise_ell=None,
                    b_ell=None):
    '''
    Combine signal and noise power spectra into total inverse
    isotropic covariance: S^-1 (S^-1 + B N^-1 B)^-1 B N^-1 B
    = (S + B^-1 N B^-1)^-1.

    Parameters
    ----------
    icov_signal_ell : (npol, npol, nell) array
        Inverse signal covariance
    icov_noise_ell : (npol, npol, nell) array
        Inverse noise covariance matrix
    b_ell : (npol, nell) array
        Beam transfer function.

    Returns
    -------
    itotcov_ell : (npol, npol, nell) array
        Total inverse covariance matrix.
    '''

    if icov_noise_ell is None:
        return icov_signal_ell.copy()

    if b_ell is not None:
        b_ell = b_ell * np.eye(b_ell.shape[0])[:,:,np.newaxis]
        in_mat = np.einsum(
            'ijl, jkl, kol -> iol', b_ell, icov_noise_ell, b_ell)
    else:
        in_mat = icov_noise_ell

    imat = mat_utils.matpow(icov_signal_ell + in_mat, -1)
    itotcov_ell = np.einsum('ijl, jkl -> ikl', imat, in_mat)
    itotcov_ell = np.einsum('ijl, jkl -> ikl', icov_signal_ell, itotcov_ell)

    return itotcov_ell

def compute_icov_alm_iso(alm, ainfo, itotcov_ell, b_ell=None, inplace=False):
    '''
    Inverse-covariance weight a set of spherical harmonic coefficients assuming
    isotropic covariance, see `get_itotcov_ell`.

    Parameters
    ----------
    alm : (npol, nelem) complex array
        Input alms.
    ainfo : curvedsky.alm_info object.
        Metainfo alms.
    itotcov_ell : (npol, npol, nell) array
        Isotropic total inverse covariance.
    b_ell : (npol, nell) array, optional
        Beam transfer function.
    inplace : bool, optional
        If set, do in-place filtering.

    Returns
    -------
    alm : (npol, nelem) complex array
        Inverse-covariance filtered alms.
    '''

    if b_ell is not None:
        # Deconvolve beam because itotcov_ell already is
        # S^-1(S^-1 + B N^-1 B)^-1 B N^-1 B.
        b_ell = b_ell * np.eye(b_ell.shape[0])[:,:,np.newaxis]
        ibell = mat_utils.matpow(b_ell, -1)
        alm_c_utils.lmul(alm, ibell, ainfo, inplace=inplace)

    alm = alm_c_utils.lmul(alm, itotcov_ell, ainfo, inplace=inplace)

    return alm

def compute_icov_iso(imap, minfo, ainfo, spin, itotcov_ell, b_ell=None):
    '''
    Inverse-covariance weight an input map assuming isotropic covariance,
    see `get_itotcov_ell`.

    Parameters
    ----------
    imap : (npol, npix) array
        Input map.
    minfo : optweight.map_utils.MapInfo object
        Metainfo input map.
    ainfo : curvedsky.alm_info object.
        Metainfo alms.
    spin : int or array-like
        Spin values for spherical harmonic transforms, should match npol.
    itotcov_ell : (npol, npol, nell) array
        Isotropic total inverse covariance.
    b_ell : (npol, nell) array, optional
        Beam transfer function.
    inplace : bool, optional
        If set, do in-place filtering.

    Returns
    -------
    alm : (npol, nelem) complex array
        Inverse-covariance filtered alms.
    '''

    npol = imap.shape[0]
    alm = np.zeros(
        (npol, ainfo.nelem), type_utils.to_complex(imap.dtype))
    sht.map2alm(imap, alm, minfo, ainfo, spin)

    return compute_icov_alm_iso(alm, ainfo, itotcov_ell, b_ell=b_ell, inplace=True)

def write_fnl(ofile, idxs, fnls, cubics, lin_terms, fishers):
    '''
    Write results from estimate_fnl to text file.

    Parameters
    ----------
    ofile : str
        Path to output txt file. Will be overwritten!
    idxs : (niter) int array
        Index for each estimate.
    fnls : (niter) array
        fNL estimates.
    cubics : (niter) array
        Cubic term for each estimate.
    lin_terms : (niter) array
        Linear term for each estimate.
    fishers : float or (niter) array
        Fisher information. For debuggin purpose, is allowed to vary per estimate.
    '''

    idxs = np.asarray(idxs)
    fnls = np.asarray(fnls)
    cubics = np.asarray(cubics)
    lin_terms = np.asarray(lin_terms)
    fishers = np.asarray(fishers)

    niter = idxs.size

    if lin_terms.size == 1:
        lin_terms *= np.ones(niter)

    if fishers.size == 1:
        fishers *= np.ones(fishers)

    mat2save = np.zeros((niter, 5))
    mat2save[:,0] = idxs
    mat2save[:,1] = fnls
    mat2save[:,2] = cubics
    mat2save[:,3] = lin_terms
    mat2save[:,4] = fishers

    header = f'fnl_mean = {np.mean(fnls):+.17e}, fnl_std = {np.std(fnls):+.17e}\n'
    header += '{:8s}\t{:24s}\t{:24s}\t{:24s}\t{:24s}'.format(
        'idx', 'fnl', 'cubic', 'lin_term', 'fisher')

    fmt = ['%9d', '%+.17e', '%+.17e','%+.17e', '%+.17e']
    np.savetxt(ofile, mat2save, fmt=fmt, delimiter='\t', header=header)

def write_fisher(ofile, lmaxs, fishers):
    '''
    Write results from estimate_fisher to text file.

    Parameters
    ----------
    ofile : str
        Path to output txt file. Will be overwritten!
    lmaxs : (niter) int array
        Lmax values.
    fishers : (niter) array
        Fisher information. For debuggin purpose, is allowed to vary per estimate.
    '''

    lmaxs = np.asarray(lmaxs)
    fishers = np.asarray(fishers)

    niter = lmaxs.size

    mat2save = np.zeros((niter, 2))
    mat2save[:,0] = lmaxs
    mat2save[:,1] = fishers

    header = '{:8s}\t{:24s}'.format(
        'lmax', 'fisher')

    fmt = ['%9d', '%+.17e']
    np.savetxt(ofile, mat2save, fmt=fmt, delimiter='\t', header=header)
    
def load_alm(ipath, pslice, lmax=None, dtype=np.complex128):
    '''
    Load alms from disk.

    Parameters
    ----------
    ipath : str
        Path to .fits file.
    pslice : slice
        Slice into first dimension of alms.
    lmax : int, optional
        Truncate up to this lmax.
    dtype : type, optional
        Cast loaded file to this dtype.

    Returns
    -------
    alm : (npol, nelem) complex array
        Loaded alms.
    ainfo : curvedsky.alm_info object
        Metainfo alms.
    '''

    hdu = np.arange(1, 100)[pslice]
    alm = np.asarray(hp.read_alm(ipath, hdu=hdu))

    ainfo = curvedsky.alm_info(hp.Alm.getlmax(alm.shape[-1]))

    if lmax:
        alm, ainfo = alm_utils.trunc_alm(alm, ainfo, lmax)

    alm = alm.astype(dtype, copy=False)

    return alm, ainfo
