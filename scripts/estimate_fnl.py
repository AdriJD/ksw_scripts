# Same as estimate_mc but start from reading state
# have option to load single map, load several maps or generate maps.
import os
import argparse

import numpy as np
from mpi4py import MPI
from optweight import map_utils, operators, wavtrans, mat_utils, fkernel, lensing
from pixell import enmap, wcsutils, curvedsky
import ksw

from ksw_scripts import script_utils

comm = MPI.COMM_WORLD
opj = os.path.join

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Estimate fnl for a set of input maps using a previously '\
        'computed state file containing the necessary Monte Carlo quantities.')

    # IO.
    parser.add_argument("--imap_files", type=str, nargs='+',
        help='Input maps from which fnl is estimated.')    
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
    parser.add_argument("--mask-file", required=True, type=str,
        help='Path to boolean mask (True for good data), either T or TQU.')
    parser.add_argument("--icov-pix-file", type=str,
        help='Path to per-pixel inverse covariance enmap .fits file')
    parser.add_argument("--cov-wav-file", type=str,
        help='Path to wavelet covariance .hdf5 file.')
    parser.add_argument("--fkernelset-file",
        help='Path to .hdf5 file with wavelet kernels.')
    parser.add_argument("--beam-fwhm", type=float,
        help='FWHM in arcmin used for the beam.')
    parser.add_argument("--beam-file", type=str,
        help='Path to beam .txt file. Alternative to beam-fwhm. Either T or TEB.')

    # Estimation.
    parser.add_argument("--T-only", dest='t_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--E-only", dest='e_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--iso-weight", action='store_true',
        help='Use isotropic icov weighting instead of CG.')
    parser.add_argument("--single", action='store_true',
        help='Use single precision.')

    # Filtering.
    parser.add_argument("--optweight-niter-cg", type=int, default=5,
        help='Number of steps with Conjugate-Gradient-based preconditioner.')
    parser.add_argument("--optweight-niter-mg", type=int, default=15,
        help='Number of steps with Multigrid-based preconditioner.')
    parser.add_argument("--optweight-spin", type=int, nargs='+',
        help='Spin values used for optweight')
    parser.add_argument("--optweight-swap-bm", action='store_true',
        help='Swap mask and beam operations. Helps convergence with large beams.')
    parser.add_argument("--optweight-scale-a", action='store_true',
        help='Rescale linear system based on beam. Helps convergence.')
    parser.add_argument("--optweight-2level-cg", type=str,
        help='Use 2-level CG-based preconditioner, either "ADEF-1" or "ADEF-2"')
    parser.add_argument("--optweight-2level-mg", type=str,
        help='Use 2-level MG-based preconditioner, either "ADEF-1" or "ADEF-2"')
    parser.add_argument("--optweight-no-masked-prec", action='store_true',
        help='Do not use preconditioners for masked pixels, used for full sky data.')
    parser.add_argument("--optweight-plm-file", type=str,
        help='Path to file containing phi_lm (and possibly omega_lm) SH coefficients')
    parser.add_argument("--optweight-plm-lmax", type=int,
        help='Custom lmax for lensing SH coefficients.')    
    parser.add_argument("--optweight-verbose", action='store_true',
        help='Print convergence to stdout')
    parser.add_argument("--optweight-save-wiener", action='store_true',
        help='Save wiener filtered alms from each rank for debugging.')
    
    # KSW.
    parser.add_argument("--ksw-theta-batch", type=int, default=100,
        help='Number of theta rings processed jointly. Increase to improve '\
             'speed at the cost of higher memory consumption. Make sure this '\
             'number exceeds the number of threads.')
    parser.add_argument("--ksw-state-file", type=str, required=True,
        help="Path to state .hdf5 file used to build the normalization and linear term.")
    parser.add_argument("--ksw-verbose", action='store_true',
        help='Print feedback to stdout')
    args = parser.parse_args()

    if comm.rank == 0:
        print(args)

    imgdir = opj(args.odir, 'img')
    logdir = opj(args.odir, 'log')
    fnldir = opj(args.odir, 'fnl')
    debugdir = opj(args.odir, 'debug')

    if comm.Get_rank() == 0:
        os.makedirs(args.odir, exist_ok=True)
        os.makedirs(imgdir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(fnldir, exist_ok=True)
        os.makedirs(debugdir, exist_ok=True)        

    if args.t_only:
        pol = ['T']
        spin = 0
        iquslice = slice(0, 1, None)
        no_te = True
        if args.e_only:
            raise ValueError('Cannot have both T and E-only.')
    elif args.e_only:
        pol = ['E']
        spin = 2
        iquslice = slice(1, 3, None)
        no_te = True
    else:
        pol = ['T', 'E']
        spin = [0, 2]
        iquslice = slice(0, 3, None)
        no_te = False

    if args.single:
        dtype = np.float32
        precision = 'single'
    else:
        dtype = np.float64
        precision = 'double'

    mask = enmap.read_map(args.mask_file)
    shape, wcs = mask.shape[-2:], mask.wcs
    mask = script_utils.process_mask(mask, iquslice)
    minfo = script_utils.find_minfo(mask.shape, mask.wcs)
    mask = map_utils.view_1d(mask, minfo)
    lmax = map_utils.minfo2lmax(minfo)

    if args.beam_file is not None:
        raise NotImplementedError
    else:
        if args.beam_fwhm is None:
            fwhm = 0
        else:
            fwhm = args.beam_fwhm
        b_ell = script_utils.get_b_ell(
            fwhm, lmax, iquslice, dtype=dtype)
        
    ainfo = curvedsky.alm_info(lmax)
    if args.signal_ps_file is not None:
        if args.signal_cov_file is not None:
            raise ValueError('Cannot have both signal ps and cov files.')
        cov_ell = script_utils.process_signal_spectra(
            np.loadtxt(args.signal_ps_file, skiprows=1, usecols=[1, 2, 3, 4]).T,
            lmax, no_te=no_te, dtype=dtype)
    elif args.signal_cov_file is not None:
        cov_ell = np.load(args.signal_cov_file)
    else:
        raise ValueError('Signal ps or cov file is needed.')
    cov_ell = script_utils.slice_spectrum(
        cov_ell, iquslice, lmax=lmax, lmin=2)
    sqrt_cov_ell_op = operators.EllMatVecAlm(
        ainfo, cov_ell, power=0.5)
    icov_ell = mat_utils.matpow(cov_ell, -1)

    if args.optweight_plm_file is not None:
        plm, ainfo_lens = script_utils.load_alm(
            args.optweight_plm_file, slice(0, 1), lmax=args.optweight_plm_lmax)
        lensop = lensing.LensAlm(plm, ainfo_lens, ainfo)
    else:
        lensop = None
    
    if args.noise_cov_file:
        cov_noise_ell = np.load(args.noise_cov_file)
        cov_noise_ell = script_utils.slice_spectrum(
            cov_noise_ell, iquslice, lmax=lmax)
        icov_noise_ell = mat_utils.matpow(cov_noise_ell, -1)
    else:
        icov_noise_ell = None

    if args.cov_wav_file and args.icov_pix_file:
        raise ValueError(f'Cannot have both cov_wav and icov_pix.')

    if args.cov_wav_file:

        cov_wav, extra = wavtrans.read_wav(
            args.cov_wav_file, extra=['nl2d'])
        nl2d = extra['nl2d']
        cov_wav, nl2d = script_utils.process_cov_wav(
            cov_wav, nl2d, iquslice, dtype=dtype)
        fkernels = fkernel.FKernelSet.from_hdf(fkernelset_file)
        sqrt_cov_wav_op = operators.WavMatVecWav(
            cov_wav, power=0.5, inplace=True)
        sqrt_n_op = operators.FMatVecF(nl2d, power=0.5)

        wav_noise_opts = dict(sqrt_cov_wav_op=sqrt_cov_wav_op,
                              sqrt_n_op=sqrt_n_op,
                              fkernels=fkernels)

        solver, prec_base, prec_masked_cg, prec_masked_mg = script_utils.init_solver(
            ainfo, minfo, icov_ell, b_ell, mask, spin,
            cov_wav=cov_wav, fkernels=fkernels, cov_noise_2d=nl2d,
            itau_ell=icov_noise_ell, swap_bm=args.optweight_swap_bm,
            scale_a=args.optweight_scale_a, lensop=lensop,
            no_masked_prec=args.optweight_no_masked_prec)

        sqrt_cov_pix_op = None

    elif args.icov_pix_file:

        icov_pix = enmap.read_map(args.icov_pix_file)
        if icov_pix.shape[-2:] != shape or not wcsutils.equal(icov_pix.wcs, wcs):
            raise ValueError(f'Mask : {shape[-2:]=}, {wcs=} '\
                             f'icov : {icov.shape[-2:]=} {icov.wcs=}')

        icov_pix = script_utils.process_icov_pix(icov_pix, iquslice, dtype=dtype)
        icov_pix = map_utils.view_1d(icov_pix, minfo)
        sqrt_cov_pix_op = operators.PixMatVecMap(
            icov_pix, power=-0.5, inplace=True)

        solver, prec_base, prec_masked_cg, prec_masked_mg = script_utils.init_solver(
            ainfo, minfo, icov_ell, b_ell, mask, spin,
            icov_pix=icov_pix, swap_bm=args.optweight_swap_bm,
            scale_a=args.optweight_scale_a, lensop=lensop,
            no_masked_prec=args.optweight_no_masked_prec)

        wav_noise_opts = {}

    icov_opts = dict(solver=solver, prec_base=prec_base,
                     prec_masked_cg=prec_masked_cg,
                     prec_masked_mg=prec_masked_mg,
                     niter_cg=args.optweight_niter_cg,
                     niter_mg=args.optweight_niter_mg,
                     two_level_cg=args.optweight_2level_cg,
                     two_level_mg=args.optweight_2level_mg,                     
                     no_masked_prec=args.optweight_no_masked_prec,
                     verbose=args.optweight_verbose)

    ############### up to here all seems the same???
    def alm_loader_template(ipath, iquslice, dtype, minfo, icov_opts, save_wiener=False):
        '''
        '''

        imap = enmap.read_map(ipath)[iquslice]
        imap = imap.astype(dtype, copy=False)
        imap = map_utils.view_1d(imap, minfo)

        if save_wiener:
            ofile = os.path.splitext(os.path.split(ipath)[-1])[0]
            ofile = opj(debugdir, f'{ofile}.fits')
            icov_opts.update(dict(ofile=ofile))
        
        return script_utils.compute_icov(imap, **icov_opts)

    alm_loader = lambda ipath : alm_loader_template(
        ipath, iquslice, dtype, minfo, icov_opts, save_wiener=args.optweight_save_wiener)

    icov = lambda alm : script_utils.compute_icov_alm(alm, iquslice, icov_opts)

    rb = ksw.ReducedBispectrum.init_from_file(args.red_bisp_file)

    estimator = ksw.KSW([rb], icov, lmax, pol, precision=precision)
    estimator.start_from_read_state(args.ksw_state_file, comm=comm)

    fisher = estimator.compute_fisher()
    fnls, cubics, lin_terms, fishers = estimator.compute_estimate_batch(
        alm_loader, args.imap_files, comm=comm, fisher=fisher,
        theta_batch=args.ksw_theta_batch, verbose=args.ksw_verbose)

    if comm.rank == 0:
        script_utils.write_fnl(opj(fnldir, 'estimates.txt'), np.arange(len(args.imap_files)),
                               fnls, cubics, lin_terms, fishers)
