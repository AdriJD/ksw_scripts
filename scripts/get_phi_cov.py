import os
import argparse

import numpy as np
import camb
from mpi4py import MPI
from ksw import Cosmology

from ksw_scripts import script_utils

comm = MPI.COMM_WORLD
opj = os.path.join

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute the real-space cov matrix of the Bardeen potential.')
    parser.add_argument("--odir", required=True,
                        help='Output directory.')
    parser.add_argument("--lmax", type=int, required=True,
                        help='Max multipole used for bispectra.')
    parser.add_argument("--rfac", type=float, default=1.,
                        help='Increase or decrease no. of radii')
    parser.add_argument("--ns", type=float, default=1.,
                        help='spectral index')
    parser.add_argument('--r-trunc', action='store_true',
                        help='Only use radii around recombination')
    parser.add_argument('--r-min', type=float,
                         help='Specify a minimum radius in Mpc.')
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    
    cosmo_opts = dict(H0=67.74, ombh2=0.02230, omch2=0.1188,
                      mnu=0.06, omk=0, tau=0.066, TCMB=2.7255)

    pars = camb.CAMBparams()
    pars.set_cosmology(**cosmo_opts)
    ip = camb.initialpower.InitialPowerLaw()
    ip.set_params(As=2.10058e-9, ns=args.ns, pivot_scalar=0.05)
    pars.set_initial_power(ip)
    cosmo = Cosmology(pars)

    if args.r_trunc:
        radii = script_utils.get_radii_rec(args.rfac, radius_min=args.r_min)
    else:
        radii = script_utils.get_radii_leo(args.rfac, radius_min=args.r_min)
    
    cov_phi = cosmo.get_real_space_phi_cov(
        radii, args.lmax, comm=comm, root=0, verbose=True)

    if comm.rank == 0:
        np.save(opj(args.odir, f'cov_phi_rfac{args.rfac:.2f}_rtrunc{int(args.r_trunc)}'),
                cov_phi)
        np.save(opj(args.odir, f'radii_rfac{args.rfac:.2f}_rtrunc{int(args.r_trunc)}'),
                radii)
