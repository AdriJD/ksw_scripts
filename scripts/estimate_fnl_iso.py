import os
import argparse

import numpy as np
from mpi4py import MPI
from optweight import map_utils, operators, wavtrans, mat_utils, alm_utils
from pixell import enmap, wcsutils, curvedsky
import healpy as hp
import ksw

from ksw_scripts import script_utils

comm = MPI.COMM_WORLD
opj = os.path.join

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Estimate fnl assuming diagonal covariance matrix. No Monte-Carlo '\
        'quantities are computed, so fast, but suboptimal in presence of mask or '\
        'anisotropic noise covariance.')
    # IO.
    parser.add_argument("--imap_files", type=str, nargs='+',
                        help='Input maps from which fnl is estimated.')
    parser.add_argument("--ialm_files", type=str, nargs='+',
                        help='Input alms from which fnl is estimated.')
    parser.add_argument("--odir", required=True, type=str,
                        help='Output directory.')    
    parser.add_argument("--red-bisp-file", type=str, required=True,
        help='Path to reduced bispectrum .hdf5 file')
    parser.add_argument("--signal-ps-file", required=True, type=str,
        help='Path to .txt file with ell, TT, EE, BB, TE columns [D_ell].')
    parser.add_argument("--noise-ps-file", type=str,
        help='Path to .npy file with (3, 3, nell) noise power spectrum [C_ell].')
    parser.add_argument("--beam-fwhm", type=float,
        help='FWHM in arcmin used for the beam.')
    parser.add_argument("--beam-file", type=str,
        help='Path to beam .txt file. Alternative to beam-fwhm.')

    # Estimation.
    parser.add_argument("--T-only", dest='t_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--E-only", dest='e_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--single", action='store_true',
        help='Use single precision.')
    parser.add_argument("--seed", default=0, type=int,
        help='Global seed from which each simulation derives its seed.')
    
    # KSW.
    parser.add_argument("--ksw-lmax", type=int,
        help='Maximum multipole used in the estimator.')
    parser.add_argument("--ksw-theta-batch", type=int, default=100,
        help='Number of theta rings processed jointly. Increase to improve '\
             'speed at the cost of higher memory consumption. Make sure this '\
             'number exceeds the number of threads.')    
    parser.add_argument("--ksw-verbose", action='store_true',
        help='Print feedback to stdout')
    args = parser.parse_args()
    
    print(args)

    imgdir = opj(args.odir, 'img')
    logdir = opj(args.odir, 'log')
    fnldir = opj(args.odir, 'fnl')

    if comm.Get_rank() == 0:
        os.makedirs(args.odir, exist_ok=True)
        os.makedirs(imgdir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(fnldir, exist_ok=True)

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
        spin = 0
        pslice = slice(0, 2, None)
        no_te = False

    if args.single:
        dtype = np.float32
        precision = 'single'
    else:
        dtype = np.float64
        precision = 'double'

    if args.imap_files is not None:
        raise NotImplementedError
        
    if args.beam_file is not None:
        raise NotImplementedError
    elif args.beam_fwhm:
        b_ell = script_utils.get_b_ell(args.beam_fwhm, args.ksw_lmax,
                          len(pol), dtype=dtype)
    else:
        b_ell = None

    ainfo = curvedsky.alm_info(args.ksw_lmax)        
    cov_ell = script_utils.process_signal_spectra(
        np.loadtxt(args.signal_ps_file, skiprows=1, usecols=[1, 2, 3, 4]).T,
        args.ksw_lmax, no_te=no_te, dtype=dtype)
    cov_ell = script_utils.slice_spectrum(cov_ell, pslice)    
    sqrt_cov_ell_op = operators.EllMatVecAlm(
        ainfo, cov_ell, power=0.5)    
    icov_ell = mat_utils.matpow(cov_ell, -1)

    if args.noise_ps_file:
        cov_noise_ell = np.load(args.noise_ps_file)
        cov_noise_ell = script_utils.slice_spectrum(cov_noise_ell, pslice)
        icov_noise_ell = mat_utils.matpow(cov_noise_ell, -1)
    else:
        icov_noise_ell = None

    def alm_loader_template(ipath, pslice, itotcov_ell, lmax, b_ell=None):
        '''
        Template function for loading alms from disk and filtering them.

        Parameters
        ----------
        ipath : str
            Path to alm file.
        pslice : slice
            Slice into alm file.
        itotcov_ell : (npol, npol, nell) array
            Isotropic inverse covariance matrix.
        lmax : int
            Truncate laoded alms to this lmax.
        b_ell : (npol, nell) array, optional
            Beam transfer function.
        
        Returns
        -------
        alm : (npol, nelem) complex array
            Loaded, sliced and filtered alm.
        '''

        hdu = np.asarray([1, 2, 3])[pslice]
        alm = np.asarray(hp.read_alm(ipath, hdu=hdu))

        ainfo = curvedsky.alm_info(hp.Alm.getlmax(alm.shape[-1]))
        
        if lmax:
            alm, ainfo = alm_utils.trunc_alm(alm, ainfo, lmax)
        
        return script_utils.icov_alm_iso(alm, ainfo, itotcov_ell, b_ell=b_ell)

    itotcov_ell = script_utils.get_itotcov_ell(
        icov_ell, icov_noise_ell=icov_noise_ell, b_ell=b_ell)

    alm_loader = lambda ipath : alm_loader_template(
        ipath, pslice, itotcov_ell, lmax=args.ksw_lmax, b_ell=b_ell)
    icov = lambda alm : script_utils.icov_alm_iso(alm, ainfo, itotcov_ell, b_ell=b_ell)

    rb = ksw.ReducedBispectrum.init_from_file(args.red_bisp_file)

    estimator = ksw.KSW([rb], icov, args.ksw_lmax, pol, precision=precision)

    fisher = estimator.compute_fisher_isotropic(
        itotcov_ell, return_matrix=False, comm=comm)
    fnls, cubics, lin_terms, fishers = estimator.compute_estimate_batch(
        alm_loader, args.ialm_files, comm=comm, fisher=fisher,
        lin_term=0, theta_batch=args.ksw_theta_batch, verbose=args.ksw_verbose)
        
    script_utils.write_fnl(opj(fnldir, 'estimates.txt'), np.arange(len(args.ialm_files)),
                           fnls, cubics, lin_terms, fishers)
