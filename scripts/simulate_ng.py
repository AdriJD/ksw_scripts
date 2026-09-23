import os
import argparse

import numpy as np
from mpi4py import MPI
from optweight import operators, mat_utils, type_utils
import ksw
import healpy as hp
from pixell import curvedsky

from ksw_scripts import script_utils

comm = MPI.COMM_WORLD
opj = os.path.join

def remove_existing(sidxs, file_template):
    '''
    Given sim indices, remove those that are already on disk.

    Parameters
    ----------
    sidxs : (nsim) int array
        Sim indices
    file_template : str
        File template containing {idx} wildcard.

    Returns
    -------
    sidxs_trunc : (nsim_trunc) int array
        Truncated sim indices array.
    '''

    sidxs_trunc = list(sidxs.copy())

    for idx, sidx in enumerate(sidxs):
        if os.path.isfile(file_template.format(idx=sidx)):
            sidxs_trunc[idx] = None
            print(f'skipping {sidx}, {opath=} already exists')
        else:
            print(f'keeping {sidx}, {opath=} was not found.')

    return np.asarray([s for s in sidxs_trunc if s is not None])

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Generate Gaussian and non-Gaussian simulations..')
    # IO.
    parser.add_argument("--odir", required=True, type=str,
        help='Output directory.')    
    parser.add_argument("--red-bisp-file", type=str, required=True,
        help='Path to reduced bispectrum .hdf5 file')
    parser.add_argument("--ialm-files", type=str, nargs='+', required=False,
        help='Input Gaussian alms. Alternative to "ksw-nsims" option')
    parser.add_argument("--signal-ps-file", required=True, type=str,
        help='Path to .txt file with ell, TT, EE, BB, TE columns [D_ell].')
    parser.add_argument("--cont", action='store_true',
        help='Do not simulate maps that are already found on disk.')
    parser.add_argument("--no-write-g", action='store_true',
        help='Do not write the Gaussian alm to disk')

    # Simulations.
    parser.add_argument("--T-only", dest='t_only', action='store_true',
        help='Only use temperature data.')
    parser.add_argument("--E-only", dest='e_only', action='store_true',
        help='Only use E-mode data.')
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
    parser.add_argument("--ksw-nsims", type=int, required=False,
        help='Number of simulations generated.')
    parser.add_argument("--ksw-sim-idx-start", type=int, default=0,
        help='Start at this sim index.')            
    parser.add_argument("--ksw-verbose", action='store_true',
        help='Print feedback to stdout')
    args = parser.parse_args()

    if comm.rank == 0:
        print(args)

    if (args.ksw_nsims is None) == (args.ialm_files is None):
        raise ValueError('Use either "ksw-nsims" or "ialm-files"')
        
    logdir = opj(args.odir, 'log')
    simdir = opj(args.odir, 'sims')

    if comm.Get_rank() == 0:
        os.makedirs(args.odir, exist_ok=True)
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(simdir, exist_ok=True)

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

    ainfo = curvedsky.alm_info(args.ksw_lmax)        
    cov_ell = script_utils.process_signal_spectra(
        np.loadtxt(args.signal_ps_file, skiprows=1, usecols=[1, 2, 3, 4]).T,
        args.ksw_lmax, no_te=no_te, dtype=dtype)
    cov_ell = script_utils.slice_spectrum(cov_ell, pslice)
    sqrt_cov_ell_op = operators.EllMatVecAlm(
        ainfo, cov_ell, power=0.5)    
    icov_ell_op = operators.EllMatVecAlm(
        ainfo, cov_ell, power=-1)    

    opath_g_template = opj(simdir, 'alm_g_{idx}.fits')
    opath_ng_template = opj(simdir, 'alm_ng_{idx}.fits')

    def alm_writer(idx, alm, opath_template):        
        '''
        Template function for writing alms to disk.

        Parameters
        ----------
        idx : int
            Sim index.
        alm : (npol, nelem) complex array
            Alm array to write.
        opath_template : str
            Output path and file containing {idx} wildcard.
        '''
        
        # Write to disk based on seed
        oname = opath_template.format(idx=idx)        
        hp.write_alm(oname, alm, overwrite=True)
    
    def alm_generator(rng):
        '''
        Template function for generating signal-only, inverse-covariance
        filtered alms.

        Parameters
        ----------
        rng : random._generator.Generator object
            Random number generator for this realization.

        Returns
        -------
        alm : (npol, nelem) complex array
            Generated alm.
        '''
        draw = script_utils.draw_signal_alm(
            sqrt_cov_ell_op, ainfo, rng, type_utils.to_complex(dtype))

        if not args.no_write_g:
            # Write to disk based on seed
            alm_writer(rng, draw, opath_g_template)
        
        return icov_ell_op(draw)

    def alm_loader(idx):
        '''
        Load and inverse-covariance filter an alm array from disk.

        Parameters
        ----------
        idx : int
            Index into `ialm_files` list.

        Returns
        -------
        alm : (npol, nelem) complex array
            Loaded and filtered alm.        
        '''

        alm, _ = script_utils.load_alm(
            args.ialm_files[idx], pslice, lmax=args.ksw_lmax,
            dtype=type_utils.to_complex(dtype))

        return icov_ell_op(alm)

    if args.ksw_nsims is not None: 
        writer = lambda rng, alm: alm_writer(rng.spawn_key[0], alm, opath_ng_template)
        loader = alm_generator
        seeds = np.random.SeedSequence(args.seed).spawn(
            args.ksw_nsims + args.ksw_sim_idx_start)
        sidxs = np.arange(len(seeds))
    else:
        # Using ialm_files instead.
        writer = lambda idx, alm: alm_writer(idx, alm, opath_ng_template)
        loader = alm_loader
        nsim = len(args.ialm_files)        
        sidxs = np.arange(nsim)
    
    rb = ksw.ReducedBispectrum.init_from_file(args.red_bisp_file)

    estimator = ksw.KSW([rb], None, args.ksw_lmax, pol, precision=precision)                       
    
    if args.cont:
        if comm.rank == 0:
            sidxs = remove_existing(sidxs, opath_ng_template)
        else:
            sidxs = None
        sidxs = ksw.utils.bcast_array(sidxs, comm)
        
    if args.ksw_nsims is not None: 
        seeds = [seeds[sidx] for sidx in sidxs]
    else:
        seeds = sidxs
        
    estimator.compute_ng_sim_batch(
        loader, seeds, writer, seeds, comm=comm, verbose=True,
        theta_batch=args.ksw_theta_batch)
