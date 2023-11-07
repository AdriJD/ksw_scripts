'''
A collection of utilities shared between scripts.
'''
import numpy as np

import healpy as hp
from optweight import map_utils, mat_utils, alm_utils, alm_c_utils, sht

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

def process_mask(mask, npol):
    '''
    Convert input mask to boolean mask.

    Parameters
    ----------
    mask_file : (npol, ny, nx) or (ny, nx) enmap
        Input mask.
    npol : int
        Number of polarizations.

    Returns
    -------
    mask_bool : (npol, npix)
        Boolean mask, True for good pixels.

    Raises
    ------
    ValueError
        If input map does not match with input npol.
    '''

    mask = enmap.read_map(mask_file)
    mask[mask < 1e-5] = 0
    mask[mask >= 1e-5] = 1
    mask = mask.astype(bool)

    mask = mat_utils.atleast_nd(mask, 3)
    if mask.shape[0] == 1:
        mask = mask * np.ones(npol)[:,np.newaxis,np.newaxis]
    elif mask.shape[0] != npol:
        raise ValueError(f'{mask.shape=} incompatible with {npol=}')

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
    w3 /= np.pi / 4.

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

def get_b_ell(fwhm, lmax, npol, dtype=np.float64):
    """
    Get beam window function given FWHM.

    Parameters
    ----------
    fwhm : float
        FWHM in arcmin.
    lmax : int
        Max multipole.
    npol : int
        Number of polarizations.
    dtype : type, optional
        Type of output.

    Returns
    -------
    b_ell : (npol, lmax + 1) array
        Gaussian beam window function.
    """

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

def slice_spectrum(cov_ell, pslice):
    '''
    Slice spectrum to desired polarizations.

    Parameters
    ----------
    cov_ell : (npol_in, npol_in, nell) array
        Input spectrum.
    pslice : slice, optional
        Slice into T, E, B axis.

    Returns
    -------
    cov_ell_out : (npol, npol, nell) array
        Sliced copy of input spectrum.
    '''

    return np.ascontiguousarray(cov_ell[pslice,pslice])

def process_cov_wav(cov_wav, nl2d, pslice, dtype=np.float64):
    '''
    Process wavelet covariance.

    Parameters
    ----------
    cov_wav : wavtrans.Wav object
        Input covariance matrix.
    nl2d : (npol, npol, nly, nlx) array
        Noise power specrum.
    pslice : slice, optional
        Slice into T, E, B axis.
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
        cov_wav.maps[key] = cov_wav.maps[key][pslice,pslice]
        # Always make copy to ensure contiguous array.
        cov_wav.maps[key] = cov_wav.maps[key].astype(dtype, copy=True)
    cov_wav.preshape = cov_wav.maps[0,0].shape[:-1]
    cov_wav.dtype = dtype

    nl2d = nl2d[pslice,pslice,:,:]
    nl2d = nl2d.astype(dtype, copy=True)

    return cov_wav, nl2d

def process_icov_pix(icov_pix, pslice, dtype=np.float64):
    '''
    If needed, slice into icov matrix. Ensure dtype.

    Parameters
    ----------
    icov_pix : (npol_in, npol_in, ny, nx) or (npol, ny, nx) enmap
        Per-pixel inverse covariance matrix.
    npol : int
        Number of polarizations.
    pslice : slice, optional
        Slice into T, E, B axis.

    Returns
    -------
    icov_pix : (npol_in, npol_in, ny, nx) or (npol, ny, nx) enmap  
        Sliced and recasted inverse covariance matrix.
    '''

    icov_pix = mat_utils.atleast_nd(icov_pix, 3)

    if icov_pix.ndim == 3:
        icov_pix = icov_pix[pslice].astype(dtype, copy=True)
    elif icov_pix.ndim == 4:
        icov_pix = icov_pix[pslice,pslice].astype(dtype, copy=True)
    else:
        raise ValueError(f'{icov_pix.shape=} not supported')

    return icov_pix

def init_solver(imap_template, ainfo, minfo, icov_ell, b_ell, mask,
                spin, icov_pix=None, cov_wav=None, fkernels=None,
                cov_noise_2d=None, itau_ell=None, swap_bm=False,
                scale_a=False):
    '''
    Initialize CG solver and preconditioners.

    Parameters
    ----------
    imap_template : (npol, npix) array
        Input map
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

    Returns
    -------
    solver :
    prec_base
    prec_masked_cg
    prec_masked_mg
    '''

    if scale_a:
        sfilt = mat_utils.matpow(b_ell, -0.5)
        lmax_mg = 3000
    else:
        sfilt = None
        lmax_mg = 6000

    if icov_pix:
        solver = solvers.from_arrays(
            imap_template, minfo, ainfo, icov_ell, icov_pix,
            b_ell=b_ell, mask_pix=mask, minfo_mask=minfo,
            draw_constr=False, spin=spin, swap_bm=swap_bm, sfilt=sfilt)

        prec_base = preconditioners.PseudoInvPreconditioner(
            ainfo, icov_ell, icov_pix, minfo, spin, b_ell=b_ell, sfilt=sfilt)

    elif icov_wav:
        solver = solvers.CGWienerMap.from_arrays_fwav(
            imap_template, minfo, ainfo, icov_ell, cov_wav, fkernels,
            b_ell=b_ell, mask_pix=mask, minfo_mask=minfo,
            draw_constr=False, spin=spin, swap_bm=swap_bm, sfilt=sfilt,
            cov_noise_2d=cov_noise_2d)

        prec_base = preconditioners.HarmonicPreconditioner(
            ainfo, icov_ell, b_ell=b_ell, itau=itau_ell, sfilt=sfilt)

    prec_masked_cg = preconditioners.MaskedPreconditionerCG(
        ainfo, icov_ell, 0, mask.astype(bool), minfo, lmax=None,
        nsteps=15, lmax_r_ell=None, sfilt=sfilt)

    prec_masked_mg = preconditioners.MaskedPreconditioner(
        ainfo, icov_ell[0:1,0:1], 0, mask[0].astype(bool), minfo,
        min_pix=1000, n_jacobi=1, lmax_r_ell=lmax_mg, sfilt=sfilt)

    return solver, prec_base, prec_masked_cg, prec_masked_mg

def icov_pix(imap, solver=None, prec_base=None, prec_masked_cg=None,
             prec_masked_mg=None, niter_cg=None, niter_mg=None,
             ofile_template=None,):
    #save_wiener=False, opath=None, write_counter=None):
    '''
    Filter input map using CG solver.

    Parameters
    ----------

    ofile_template : str
        If provided, write Wiener-filtered map to disk. May contain
        {idx} in path of filename, e.g. "/path/to/sim_{idx}.fits".

    Returns
    -------

    '''

    solver.reset_preconditioner()
    solver.set_b_vec(imap)
    solver.add_preconditioner(prec_base)
    solver.add_preconditioner(prec_masked_cg)
    solver.init_solver()

    niter = niter_cg + niter_mg

    for idx in range(niter_cg):
        solver.step()
        print(solver.i, solver.err)

    solver.reset_preconditioner()
    solver.add_preconditioner(prec_base)
    solver.add_preconditioner(prec_masked_mg, sel=np.s_[0])

    solver.b_vec = solver.b0
    if niter_cg == 0:
        solver.init_solver(x0=None)
    else:
        solver.init_solver(x0=solver.x)

    for idx in range(niter_cg, niter):

        solver.step()
        print(solver.i, solver.err, f'rank : {comm.rank}')

    if save_wiener:
        filename = opj(
            opath, f'mc_gt_w_{write_counter[0]}.fits')
        write_counter[0] += 1
        hp.write_alm(opj(opath, filename), solver.x, overwrite=True)

    return solver.get_icov()

def draw_noise_wav(minfo, seed, dtype=np.float64, sqrt_cov_wav_op=None,
                   sqrt_n_op=None, fkernels=None):
    '''

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

    '''

    rng = np.random.default_rng(seed)

    sqrt_cov_pix_op.mpix.shape[0]
    omap = rng.standard_normal(
        npol * minfo.npix).reshape((npol, minfo.npix))
    omap = sqrt_cov_pix_op(omap)

    return omap

def draw_signal_alm(sqrt_cov_ell_op, ainfo, seed, dtype):
    '''

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
    Generate signal + noise simulation and return the
    inverse-covariance filtered version.

    Parameters
    ----------
    seed :
    sqrt_cov_ell_op
    b_ell
    minfo
    ainfo
    spin
    mask
    dtype
    sqrt_cov_pix_op
    noise_wav_ops
    icov_ops
    '''

    rng = np.random.default_rng(seed)

    alm = draw_signal_alm(
        sqrt_cov_ell_op, ainfo, rng, type_utils.to_complex(dtype))

    alm_c_utils.lmul(alm, b_ell, ainfo, inplace=True)
    omap = np.zeros((alm.shape[0], minfo.npix), dtype=dtype)
    sht.alm2map(alm, omap, ainfo, minfo, spin)

    # Only mask signal.
    omap *= mask

    if noise_wav_opts:
        omap += draw_noise_wav(minfo, rng, dtype=dtype,
                               **wav_noise_opts)
    else:
        omap += draw_noise_pix(sqrt_cov_pix_op, minfo, rng, dtype)

    return icov_pix(omap, **icov_opts)

def icov_alm(alm, icov_opts):
    '''
    Inverse-covariance filter a set of alm coefficients by
    first applying the P projection matrix (d_pix = P slm + n).

    Parameters
    ----------
    alm : (npol, nelem) complex array
        Input alms.
    icov_opts : dict
        Keyword arguments for `icov_pix`.

    Returns
    -------
    oalm : (npol, nelem)
        Inverse-covariance filtered alms.
    '''

    omap = solver.proj(alm)

    return icov_pix(omap, **icov_opts)

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

    icov_noise_ell = mat_utils.matpow(n_ell, -1)

    if b_ell is not None:
        b_ell = b_ell * np.eye(b_ell.shape[0])[:,:,np.newaxis]
        in_mat = np.einsum(
            'ijl, jkl, kol -> iol', b_ell, icov_noise_ell, b_ell)
    else:
        in_mat = icov_noise_ell

    imat = mat_utils.matpow(icov_ell + in_mat, -1)
    itotcov_ell = np.einsum('ijl, jkl -> ikl', imat, in_mat)
    itotcov_ell = np.einsum('ijl, jkl -> ikl', icov_ell, itotcov_ell)
    
    return itotcov_ell

def icov_alm_iso(alm, ainfo, itotcov_ell, b_ell=None, inplace=False):
    '''

    '''

    if b_ell is not None:
        # Deconvolve beam because itotcov_ell already is
        # S^-1(S^-1 + bN^-1b)^-1bN^-1b
        b_ell = b_ell * np.eye(b_ell.shape[0])[:,:,np.newaxis]
        ibell = mat_utils.matpow(b_ell, -1)
        alm_c_utils.lmul(alm, ibell, ainfo, inplace=inplace)    

    alm = alm_c_utils.lmul(alm, itotcov_ell, ainfo, inplace=inplace)

    return alm

def icov_pix_iso(imap, minfo, ainfo, spin, itotcov_ell, b_ell=None):
    '''

    '''

    npol = imap.shape[0]
    alm = np.zeros(
        (npol, ainfo.nelem), type_utils.to_complex(imap.dtype))
    sht.map2alm(imap, alm, minfo, ainfo, spin)

    return icov_alm_iso(alm, ainfo, itotcov_ell, b_ell=b_ell, inplace=True)
