import os
import argparse

import numpy as np
from mpi4py import MPI
from optweight import map_utils, operators, wavtrans, mat_utils, alm_utils, type_utils
from pixell import enmap, wcsutils, curvedsky
import healpy as hp
import ksw

from ksw_scripts import script_utils

comm = MPI.COMM_WORLD
opj = os.path.join

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Estimate fisher assuming diagonal covariance matrix. No Monte-Carlo '\
        'quantities are computed, so fast, but unrealistic presence of mask or '\
        'anisotropic noise covariance.')
    # IO.
    parser.add_argument("--odir", required=True, type=str,
        help='Output directory.')    
    parser.add_argument("--red-bisp-file", type=str, required=True,
        help='Path to reduced bispectrum .hdf5 file')
    parser.add_argument("--signal-ps-file", type=str,
        help='Path to .txt file with ell, TT, EE, BB, TE columns [D_ell].')
    parser.add_argument("--signal-cov-file", type=str,
        help='Path to .npy file with (3, 3, nell) signal power spectrum [C_ell].')    
    parser.add_argument("--noise-cov-file", type=str,
        help='Path to .npy file with (3, 3, nell) noise power spectrum [C_ell].')
    parser.add_argument("--beam-fwhm", type=float,
        help='FWHM in arcmin used for the beam.')
    parser.add_argument("--beam-file", type=str,
        help='Path to beam .txt file. Alternative to beam-fwhm.')

    # Fisher.
    parser.add_argument("--T-only", dest='t_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--E-only", dest='e_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--single", action='store_true',
        help='Use single precision.')
    
    # KSW.
    parser.add_argument("--ksw-lmax", type=int, nargs='+',
        help='Maximum multipole used in the estimator.')
    parser.add_argument("--ksw-verbose", action='store_true',
        help='Print feedback to stdout')
    args = parser.parse_args()

    if comm.rank == 0:
        print(args)

    imgdir = opj(args.odir, 'img')
    logdir = opj(args.odir, 'log')
    fnldir = opj(args.odir, 'fnl')

    if comm.rank == 0:
        os.makedirs(args.odir, exist_ok=True)
        os.makedirs(imgdir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(fnldir, exist_ok=True)
    comm.Barrier()
        
    if args.t_only:
        pol = ['T']
        pslice = slice(0, 1, None)
        no_te = True
        if args.e_only:
            raise ValueError('Cannot have both T and E-only.')
    elif args.e_only:
        pol = ['E']
        pslice = slice(1, 2, None)
        no_te = True
    else:
        pol = ['T', 'E']
        pslice = slice(0, 2, None)
        no_te = False

    if args.single:
        dtype = np.float32
        precision = 'single'
    else:
        dtype = np.float64
        precision = 'double'

    lmax_array = np.asarray(args.ksw_lmax)
    lmax_max = lmax_array.max()
        
    if args.beam_file is not None:
        raise NotImplementedError
    elif args.beam_fwhm:
        b_ell = script_utils.get_b_ell(
            args.beam_fwhm, lmax_max, pslice, dtype=dtype)
    else:
        b_ell = None

    if args.signal_ps_file is not None:
        if args.signal_cov_file is not None:
            raise ValueError('Cannot have both signal ps and cov files.')
        cov_ell = script_utils.process_signal_spectra(
            np.loadtxt(args.signal_ps_file, skiprows=1, usecols=[1, 2, 3, 4]).T,
            lmax_max, no_te=no_te, dtype=dtype)
    elif args.signal_cov_file is not None:
        cov_ell = np.load(args.signal_cov_file)
    else:
        raise ValueError('Signal ps or cov file is needed.')
    cov_ell = script_utils.slice_spectrum(
        cov_ell, pslice, lmax=lmax_max, lmin=2)
    icov_ell = mat_utils.matpow(cov_ell, -1)

    if args.noise_cov_file:
        cov_noise_ell = np.load(args.noise_cov_file)
        cov_noise_ell = script_utils.slice_spectrum(
            cov_noise_ell, pslice, lmax=lmax_max)
        icov_noise_ell = mat_utils.matpow(cov_noise_ell, -1)
    else:
        icov_noise_ell = None

    itotcov_ell = script_utils.get_itotcov_ell(
        icov_ell, icov_noise_ell=icov_noise_ell, b_ell=b_ell)

    rb = ksw.ReducedBispectrum.init_from_file(args.red_bisp_file)
    
    fisher_array = np.zeros(lmax_array.size)
    for lidx, lmax in enumerate(lmax_array):
    
        estimator = ksw.KSW([rb], None, lmax, pol, precision=precision)
        itotcov_ell_trunc = np.ascontiguousarray(itotcov_ell[:,:,:lmax+1])
        fisher_array[lidx] = estimator.compute_fisher_isotropic(
            itotcov_ell_trunc, return_matrix=False, comm=comm)

    if comm.rank == 0:
        script_utils.write_fisher(
            opj(fnldir, 'fisher_estimates.txt'), lmax_array, fisher_array)
            
