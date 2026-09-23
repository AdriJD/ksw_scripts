import os
import argparse

import numpy as np
from scipy.interpolate import CubicSpline
from mpi4py import MPI
from optweight import mat_utils, type_utils, alm_c_utils, alm_utils, map_utils, sht
import ksw
import healpy as hp
from pixell import curvedsky

from ksw_scripts import script_utils

comm = MPI.COMM_WORLD
opj = os.path.join

def m_op(plm, alpha, ainfo):
    '''
    Compute alm = int r^2 dr alpha(r)_ell plm(r), for input plm(r), where
    alpha(r)_ell = (2 / pi) int k^2 dk T^Phi_ell(k) j_ell(kr), with T^Phi_ell(k)
    the CMB transfer function for the Bardeen potential Phi.
    
    Parameters
    ----------
    plm : (nr, nelem) complex array
        Phi per shell.
    alpha : (nr, npol, nell) array
        The alpha(r) array premultiplied with r^2 dr.
    ainfo : curvedsky.alm_info object
        Meta info plms.

    Returns
    -------
    alm : (npol, nelem) complex array
        CMB spherical harmonic coefficients.
    '''

    nr, npol = alpha.shape[:2]
    lmax = alpha.shape[-1] - 1
    
    assert ainfo.lmax == lmax
    assert plm.shape[:2] == (nr, ainfo.nelem), f'Mismatch {alpha.shape} and {plm.shape=}'
    
    out = np.zeros((npol, ainfo.nelem), dtype=np.complex128)
    
    for ridx in range(nr):
        for pidx in range(npol):
            out[pidx] += alm_c_utils.lmul(plm[ridx], alpha[ridx,pidx], ainfo)
            
    return out

def mt_op(alm, alpha, ainfo):
    '''
    Apply the transpose of M to an input set of alms.

    Parameters
    ----------
    alm : (npol, nelem) complex array
        CMB spherical harmonic coefficients.
    alpha : (nr, npol, nell) array
        The alpha(r) array premultiplied with r^2 dr.
    ainfo : curvedsky.alm_info object
        Meta info alms.

    Returns
    -------
    out : (nr, nelem) complex array
        Result per shell.
    '''

    nr, npol = alpha.shape[:2]
    lmax = alpha.shape[-1] - 1
    
    assert ainfo.lmax == lmax
    assert alm.shape == (npol, ainfo.nelem), f'Mismatch {alpha.shape} and {alm.shape=}'
    
    out = np.zeros((nr, ainfo.nelem), dtype=np.complex128)
    
    for ridx in range(nr):
        for pidx in range(npol):
            out[ridx,:] += alm_c_utils.lmul(alm[pidx], alpha[ridx,pidx], ainfo)
            
    return out

def get_p_fluc(phi_cov, alpha, icov_ell):
    '''
    Compute the covariance matrix needed to draw constrained Phi.

    Parameters
    ----------
    phi_cov : (nr, nr, nell) array
        Phi covariance matrix.
    alpha : (nr, npol, nell) array
        The alpha(r) array premultiplied with r^2 dr.
    icov_ell : (npol, npol, nell) array
        Inverse covariance matrix of the CMB alms.

    Returns
    -------
    p_fluc : (nr, nr, nell) array
        P_fluc covariance matrix.
    '''

    nr, npol = alpha.shape[:2]
    lmax = alpha.shape[-1] - 1

    assert icov_ell.shape == (npol, npol, lmax + 1), f'{icov_ell.shape} != {alpha.shape}'
    
    mp = np.einsum('ipl, ijl -> jpl', alpha, phi_cov)
    p2 = np.einsum('ipl, pql, jql -> ijl', mp, icov_ell, mp)
    
    return phi_cov - p2    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Given set of Gaussian CMB fields, generate the local \
        non-Gaussian contribution.')
    # IO.
    parser.add_argument("--odir", required=True, type=str,
        help='Output directory.')
    parser.add_argument("--ialm-files", type=str, nargs='+', required=True,
        help='Input Gaussian alms.')    
    parser.add_argument("--red-bisp-file", type=str, required=True,
        help='Path to local reduced bispectrum .hdf5 file. Only alpha(r) is used.')
    parser.add_argument("--radii-rb-file", type=str, required=True,
        help='Path to radii corresponding to the reduced bispectrum file.')    
    parser.add_argument("--signal-ps-file", type=str,
        help='Path to .txt file with ell, TT, EE, BB, TE columns [D_ell].')
    parser.add_argument("--signal-cov-file", type=str,
        help='Path to .npy file with (3, 3, nell) signal power spectrum [C_ell].')    
    parser.add_argument("--cov-phi-file", required=True, type=str,
        help='Path to .npy file with Phi covariance, see `get_phi_cov.py`.')
    parser.add_argument("--radii-phi-file", type=str, required=True,
        help='Path to radii corresponding to the phi covariance file.')    
    parser.add_argument("--cont", action='store_true',
        help='Do not simulate maps that are already found on disk.')
    parser.add_argument("--write-phi", action='store_true',
        help='Write the MAP estimate Phi to disk.')
    parser.add_argument("--write-phi-constr", action='store_true',
        help='Write the constrained realization of Phi to disk.')    
    parser.add_argument("--single", action='store_true',
        help='Output in single precision.')

    # Simulations.
    parser.add_argument("--T-only", dest='t_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--E-only", dest='e_only', action='store_true',
        help='Only use E-mode data.')
    parser.add_argument("--seed", default=0, type=int,
        help='Global seed from which each simulation derives its seed.')
    
    parser.add_argument("--lmax", type=int,
        help='Maximum multipole used for the output. Note, strictly speaking, the \
        input Gaussian alms should be at 2 * lmax.')
    args = parser.parse_args()

    if comm.rank == 0:
        print(args)

    simdir = opj(args.odir, 'sims')

    if comm.Get_rank() == 0:
        os.makedirs(args.odir, exist_ok=True)
        os.makedirs(simdir, exist_ok=True)

    if args.t_only:
        pol = ['T']
        spin = 0
        pslice = slice(0, 1, None)
        no_te = True
        if args.e_only:
            raise ValueError('Cannot have both T and E-only.')
    elif args.e_only:
        pol = ['E']
        spin = 0
        pslice = slice(1, 2, None)
        no_te = True
    else:
        pol = ['T', 'E']
        spin = [0, 2]
        pslice = slice(0, 2, None)
        no_te = False

    if args.single:
        dtype = np.float32
    else:
        dtype = np.float64

    ainfo = curvedsky.alm_info(args.lmax)
    if args.signal_ps_file is not None:
        if args.signal_cov_file is not None:
            raise ValueError('Cannot have both signal ps and cov files.')
        cov_ell = script_utils.process_signal_spectra(
            np.loadtxt(args.signal_ps_file, skiprows=1, usecols=[1, 2, 3, 4]).T,
            args.lmax, no_te=no_te, dtype=dtype)
    elif args.signal_cov_file is not None:
        cov_ell = np.load(args.signal_cov_file)
    else:
        raise ValueError('Signal ps or cov file is needed.')
    cov_ell = script_utils.slice_spectrum(
        cov_ell, pslice, lmax=args.lmax, lmin=2)

    icov_ell = mat_utils.matpow(cov_ell, -1)
    del cov_ell

    # Load and symmetrize cov.
    cov_phi = np.load(args.cov_phi_file)
    cov_phi = cov_phi[:,:,:args.lmax+1]
    cov_phi = 0.5 * (cov_phi + cov_phi.transpose(1, 0, 2))    
    radii = np.load(args.radii_phi_file)
    
    rb = ksw.ReducedBispectrum.init_from_file(args.red_bisp_file)
    radii_alpha = np.load(args.radii_rb_file)

    # Shape = (ncomp, nr, npol, nell).
    alpha = rb.factors.reshape(2, radii_alpha.size, 2, rb.ells_full.size)[0]
    alpha = alpha[:,pslice,:]

    # Convert T_zeta to T_phi
    alpha *= (5 / 3) 

    # Interpolate alpha to cov_phi radii.
    cs = CubicSpline(radii_alpha, alpha, axis=0, extrapolate=False)
    del radii_alpha
    alpha = np.nan_to_num(cs(radii), nan=0.0, copy=False)
    # Add ell=0 and ell=1.
    alpha_ext = np.zeros((radii.size, len(pol), args.lmax + 1))
    alpha_ext[:,:,2:] = alpha[:,:,:args.lmax-1]
    alpha = alpha_ext

    # Include dr r^2 weights.
    wr = ksw.utils.get_trapz_weights(radii) * radii ** 2
    alpha *= wr[:,np.newaxis,np.newaxis]
    
    # Compute sqrT_P_flux
    p_fluc = get_p_fluc(cov_phi, alpha, icov_ell)    
    sqrt_p_fluc = mat_utils.matpow(p_fluc, 0.5)
    del p_fluc

    opath_ng_template = opj(simdir, 'alm_ng_{idx}.fits')
    nsim = len(args.ialm_files)        
    seeds = np.random.SeedSequence(args.seed).spawn(nsim)
    sidxs = np.arange(nsim)

    if args.cont:        
        sidxs_trunc = None
        if comm.rank == 0:
            sidxs_trunc = list(sidxs.copy())
            for idx, sidx in enumerate(sidxs):
                opath = opath_ng_template.format(idx=sidx)
                if os.path.isfile(opath):
                    sidxs_trunc[idx] = None
                    print(f'skipping {sidx}, {opath=} already exists')
                else:
                    print(f'keeping {sidx}, {opath=} was not found.')
            sidxs_trunc = np.asarray([s for s in sidxs_trunc if s is not None])
        sidxs = ksw.utils.bcast_array(sidxs_trunc, comm)
    sidxs_per_rank = sidxs[comm.rank::comm.size]
    
    for sidx in sidxs_per_rank:

        if args.write_phi or args.write_phi_constr:
            odir_phi = opj(simdir, f'phi_{sidx}')
            os.makedirs(odir_phi, exist_ok=True)
        
        rng = np.random.default_rng(seeds[sidx])
        alm, _ = script_utils.load_alm(
            args.ialm_files[sidx], pslice, lmax=args.lmax,
            dtype=type_utils.to_complex(dtype))
        
        alm = alm_c_utils.lmul(alm, icov_ell, ainfo=ainfo, inplace=True)
        plm = mt_op(alm, alpha, ainfo)
        plm = alm_c_utils.lmul(plm, cov_phi, ainfo, inplace=True)

        if args.write_phi:
            for ridx in range(plm.shape[0]):
                hp.write_alm(opj(odir_phi, f'plm_{ridx}.fits'), plm[ridx], overwrite=True)
        
        unit_var_plm = alm_utils.unit_var_alm(ainfo, (radii.size,), rng)
        # This adds the fluctuation term to plm in place.
        plm_constr = alm_c_utils.lmul(unit_var_plm, sqrt_p_fluc, ainfo,
                                      alm_out=plm, add=True)
        del unit_var_plm

        if args.write_phi_constr:
            for ridx in range(plm.shape[0]):
                hp.write_alm(opj(odir_phi, f'plm_constr_{ridx}.fits'), plm[ridx],
                             overwrite=True)

        # Square in place by looping over r.
        minfo = map_utils.MapInfo.map_info_gauss_legendre(
            2 * args.lmax + 1, 4 * args.lmax + 1)
        tmp_map = np.zeros(minfo.npix)

        # Save memory, and overwrite plm_constrained
        for ridx in range(radii.size):
            sht.alm2map(plm_constr[ridx], tmp_map, ainfo, minfo, 0)
            tmp_map **= 2
            sht.map2alm(tmp_map, plm_constr[ridx], minfo, ainfo, 0)
            plm_constr[ridx,0] = 0
        del tmp_map
        
        # Project back.
        alm_ng = m_op(plm_constr, alpha, ainfo)
        del plm_constr
        
        hp.write_alm(opj(opath_ng_template.format(idx=sidx)), alm_ng, overwrite=True)
