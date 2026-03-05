import os
import argparse

import numpy as np
from mpi4py import MPI
from optweight import map_utils, operators, wavtrans, mat_utils, fkernel, lensing, sht
from pixell import enmap, wcsutils, curvedsky
import ksw

from ksw_scripts import script_utils

comm = MPI.COMM_WORLD
opj = os.path.join

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Estimate the Monte Carlo quantities needed for '\
        'the estimator normalization and linear term.')

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
    parser.add_argument("--mask-file", required=True, type=str,
        help='Path to boolean mask (True for good data), either T or TQU. If .fits'\
              'file, assume enmap, if .hdf5 file assume Gauss-Legendre map.')
    parser.add_argument("--icov-pix-file", type=str,
        help='Path to per-pixel inverse covariance enmap .fits file. If .fits'\
             'file, assume enmap, if .hdf5 file assume Gauss-Legendre map.')
    parser.add_argument("--cov-wav-file", type=str,
        help='Path to wavelet covariance .hdf5 file.')
    parser.add_argument("--fkernelset-file",
        help='Path to .hdf5 file with wavelet kernels.')
    parser.add_argument("--beam-fwhm", type=float,
        help='FWHM in arcmin used for the beam.')
    parser.add_argument("--beam-file", type=str,
        help='Path to beam .txt file. Alternative to beam-fwhm. Either T or TEB.')
    parser.add_argument("--write-grad-t", action='store_true',
        help='If set, store the grad T alms in the debug directory.')
    parser.add_argument("--imap-files", type=str, nargs='+',
        help='Input maps to estimate Monte-Carlo quantities from. If not given, maps '\
             'will be generated from data model.')
    parser.add_argument("--imap-indices", type=int, nargs='+',
        help='Indices of input map files, see --imap-file-template.')
    parser.add_argument("--imap-file-template", type=str,
        help='Template of imap file names that can be parsed as python string '\
             'and contains "{idx}". For example: "/path/to/sim_{idx:03d}.fits"')    
    parser.add_argument("--mask-imap", action='store_true',
        help='Apply the mask to the input maps (so assume maps are unmasked).')
    
    # Estimation.
    parser.add_argument("--T-only", dest='t_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--E-only", dest='e_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--iso-weight", action='store_true',
        help='Use isotropic icov weighting instead of CG.')
    parser.add_argument("--single", action='store_true',
        help='Use single precision.')
    parser.add_argument("--seed", default=0, type=int,
        help='Global seed from which each simulation derives its seed.')

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
    parser.add_argument("--optweight-use-prec-harm", action='store_true',
        help='Use the harmonic preconditioner as the base preconditioner for '\
        'pixel-based noise model instead of the preudo-inverse preconditioner')
    parser.add_argument("--optweight-niter-noise-cg", type=int, default=6,
        help='Number of CG steps used to invert constant-correlation noise model.')
    parser.add_argument("--optweight-no-masked-noise", action='store_true',
        help='If set, assume that the noise has not been masked, i.e. M in the the noise '\
            'model is set to 1. Only relevant when constant-correlation noise model is used.')
    parser.add_argument("--optweight-no-te", action='store_true',
        help='If set, remove the TE correlation from the signal spectrum that is loaded.')
    
    # KSW.
    parser.add_argument("--ksw-niter", type=int, default=100,
        help='Number of simulations used for Monte-Carlo quantities.')
    parser.add_argument("--ksw-theta-batch", type=int, default=100,
        help='Number of theta rings processed jointly. Increase to improve '\
             'speed at the cost of higher memory consumption. Make sure this '\
             'number exceeds the number of threads.')
    parser.add_argument("--ksw-state-file", type=str,
        help="Path to state .hdf5 file to restart from.")
    parser.add_argument("--ksw-verbose", action='store_true',
        help='Print feedback to stdout')
    args = parser.parse_args()

    # ADD DEBUG OPTION TO WRITE MAPS AFTER EACH CG RUN.
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

    if args.imap_files is not None and args.imap_indices is not None:
        raise ValueError('Cannot have both --imap-files and --imap-indices')
    
    if (args.imap_indices is None) != (args.imap_file_template is None):
        raise ValueError('--imap-file-template requires --imap-indices')
    
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

    if args.optweight_no_te:
        no_te = True
        
    if args.single:
        dtype = np.float32
        precision = 'single'
    else:
        dtype = np.float64
        precision = 'double'

    try:
        mask = enmap.read_fits(args.mask_file)
    except OSError:
        mask, minfo = map_utils.read_map(args.mask_file)
    else:
        minfo = script_utils.find_minfo(mask.shape, mask.wcs)
        mask = map_utils.view_1d(mask, minfo)
    mask = script_utils.process_mask(mask, iquslice)
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

        try:
            icov_pix = enmap.read_fits(args.icov_pix_file)
        except OSError:
            icov_pix, minfo_icov = map_utils.read_map(args.icov_pix_file)
        else:
            minfo_icov = script_utils.find_minfo(icov_pix.shape, icov_pix.wcs)
            icov_pix = map_utils.view_1d(icov_pix, minfo)

        if not map_utils.minfo_is_equiv(minfo, minfo_icov):
            raise ValueError('Mask geometry does not match icov_pix geometry.')

        icov_pix = script_utils.process_icov_pix(icov_pix, iquslice, dtype=dtype)

        if icov_noise_ell is not None:            
            sqrt_cov_pix_op = None

            ell_op = operators.EllMatVecAlm(
                ainfo, icov_noise_ell, power=-0.5)
            pix_op = operators.PixMatVecMap(
                icov_pix, power=-0.5, inplace=True)            
            #sqrt_cov_noise_ell_op = lambda x: pix_op(ell_op(x))
            def sqrt_cov_noise_ell_op_full(ialm, ainfo, minfo, spin):
                oalm = ell_op(ialm)
                noise = np.zeros((ialm.shape[0], minfo.npix))
                sht.alm2map(oalm, noise, ainfo, minfo, spin)
                return pix_op(noise)
            sqrt_cov_noise_ell_op = lambda x: sqrt_cov_noise_ell_op_full(
                x, ainfo, minfo, spin)
                
        else:
            sqrt_cov_pix_op = operators.PixMatVecMap(
                icov_pix, power=-0.5, inplace=True)
            sqrt_cov_noise_ell_op = None        
        
        solver, prec_base, prec_masked_cg, prec_masked_mg = script_utils.init_solver(
            ainfo, minfo, icov_ell, b_ell, mask, spin,
            icov_pix=icov_pix, swap_bm=args.optweight_swap_bm,
            scale_a=args.optweight_scale_a, lensop=lensop,
            no_masked_prec=args.optweight_no_masked_prec,
            use_prec_harm=args.optweight_use_prec_harm,
            icov_noise_ell=icov_noise_ell, no_masked_noise=args.optweight_no_masked_noise,
            nsteps_noise_cg=args.optweight_niter_noise_cg)

        wav_noise_opts = {}

    # Unique to each rank. Just here to keep track of number of maps written.
    # THIS SHOULD BE UNIQUE NUMBER FOR EACH RANK. ALSO HOW CAN YOU DISTINGUISH
    # BETWEEN MC_GT AND DATA ALMS? IF YOU ADD A template string to dict that
    # is supposed to be the output, you could do alm_wiener_{idx} and mc_gt_wiener{idx}.
    # idx can be given by the rng entropy? or the filename?
    write_counter = np.asarray([0])

    icov_opts = dict(solver=solver, prec_base=prec_base,
                     prec_masked_cg=prec_masked_cg,
                     prec_masked_mg=prec_masked_mg,
                     niter_cg=args.optweight_niter_cg,
                     niter_mg=args.optweight_niter_mg,
                     two_level_cg=args.optweight_2level_cg,
                     two_level_mg=args.optweight_2level_mg,
                     no_masked_prec=args.optweight_no_masked_prec,
                     verbose=args.optweight_verbose)
                     #save_wiener=False, opath=None, write_counter=None)

    # Depending on whether we're loading maps from disk or generating them from the data
    # model, we need a different alm_loader.
    if (args.imap_files is not None) or (args.imap_indices is not None):
        
        def alm_loader_template(ipath, iquslice, dtype, minfo, icov_opts,
                                imap_file_template=None, mask_imap=False):
            '''
            Load up an input map (and potentially a set of lensing potential alms),
            initialize the CG solver and return inverse-covariance filtered data.

            Parameters
            ----------
            ipath : str or int
                Either a filename or an index.
            iquslice : slice
                Slice into IQU axis.
            dtype : type
                Convert loaded input to this type.
            minfo : optweight.map_utils.MapInfo object
                Metainfo mask and imap.
            icov_opts : dict
                Keyword arguments to script_utils.compute_icov.
            imap_file_template : str, optional
                Filename template, used in combination with integer `ipath`.
            mask_imap : bool, optional
                If True, apply mask to input maps.

            Returns
            -------
            icov_alm : (npol, nelem) complex array
                Spherical harmonic coefficients of the inverse-covariance filtered
                input map.        
            '''

            if isinstance(ipath, str):
                filename = ipath
            elif int(ipath) == ipath:
                # Index instead of str.
                filename = imap_file_template.format(idx=ipath)
            else:
                raise ValueError(f'{ipath=} not understood')

            print(f'Loading {filename}')        
            try:
                imap = enmap.read_fits(filename)
            except OSError:
                imap, minfo_imap = map_utils.read_map(filename)
            else:
                minfo_imap = script_utils.find_minfo(imap.shape, imap.wcs)
                imap = map_utils.view_1d(imap, minfo)

            if not map_utils.minfo_is_equiv(minfo, minfo_imap):
                raise ValueError('Mask geometry does not match imap geometry.')

            imap = imap[iquslice]
            imap = imap.astype(dtype, copy=False)

            if mask_imap:
                imap *= mask

            return script_utils.compute_icov(imap, **icov_opts)

        alm_loader = lambda ipath : alm_loader_template(
            ipath, iquslice, dtype, minfo, icov_opts,
            imap_file_template=args.imap_file_template, mask_imap=args.mask_imap)
        
    else:
        sqrt_cov_ell_op = operators.EllMatVecAlm(
            ainfo, cov_ell, power=0.5)        
        alm_loader = lambda rng : script_utils.alm_loader_template(
            rng, sqrt_cov_ell_op, script_utils.slice2len(iquslice), b_ell, minfo, ainfo, spin,
            mask, dtype, sqrt_cov_pix_op=sqrt_cov_pix_op,
            sqrt_cov_noise_ell_op=sqrt_cov_noise_ell_op,
            wav_noise_opts=wav_noise_opts, icov_opts=icov_opts)

    icov = lambda alm : script_utils.compute_icov_alm(alm, iquslice, icov_opts)

    rb = ksw.ReducedBispectrum.init_from_file(args.red_bisp_file)

    estimator = ksw.KSW([rb], icov, lmax, pol, precision=precision)
    if args.ksw_state_file is not None:
        estimator.start_from_read_state(args.ksw_state_file, comm=comm)

    if args.imap_file_template:
        seeds = args.imap_indices
    elif args.imap_files is not None:
        seeds = args.imap_files
    else:        
        seeds = np.random.SeedSequence(args.seed).spawn(args.ksw_niter + estimator.mc_idx)
        
    estimator.step_batch_2pass(
        alm_loader, seeds, comm=comm, verbose=False, theta_batch=args.ksw_theta_batch)        
    
    fisher = estimator.compute_fisher_2pass(comm)
    if comm.rank == 0:
        print(f'{fisher=}')
    estimator.write_state_2pass(
        opj(fnldir, f'state_{estimator.mc_idx}'), fisher, comm=comm)
