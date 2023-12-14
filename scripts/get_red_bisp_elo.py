'''
Compute the reduced bispectra for the Local, Equilateral
and Orthogonal models.
'''
import os
import argparse

import numpy as np
import camb
from ksw import Shape, Cosmology

from ksw_scripts import script_utils

opj = os.path.join

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Compute and store reduced bispectra.')
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
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    
    cosmo_opts = dict(H0=67.74, ombh2=0.02230, omch2=0.1188,
                      mnu=0.06, omk=0, tau=0.066, TCMB=2.7255)

    pars = camb.CAMBparams()
    pars.set_cosmology(**cosmo_opts)
    ip = camb.initialpower.InitialPowerLaw()
    # Note, ns here does not really matter, only for camb Cls.
    ip.set_params(As=2.10058e-9, ns=args.ns, pivot_scalar=0.05)
    pars.set_initial_power(ip)
    cosmo = Cosmology(pars)

    cosmo.compute_transfer(args.lmax)
    cosmo.compute_c_ell()
    cosmo.write_transfer(opj(args.odir, f'transfer_lmax{args.lmax}'))
    cosmo.write_c_ell(opj(args.odir, f'c_ell_lmax{args.lmax}'))

    local = Shape.prim_local(ns=args.ns)
    equilateral = Shape.prim_equilateral(ns=args.ns)
    orthogonal = Shape.prim_orthogonal(ns=args.ns)

    if args.r_trunc:
        radii = script_utils.get_radii_rec(args.rfac)
    else:
        radii = script_utils.get_radii_leo(args.rfac)

    for shape in [local, equilateral, orthogonal]:

        if len(cosmo.red_bispectra) > 0:
            cosmo.red_bispectra = []

        cosmo.add_prim_reduced_bispectrum(shape, radii)
        rb = cosmo.red_bispectra[0]
        rb.write(opj(args.odir, f'{shape.name}_lmax{args.lmax}_ns{args.ns:.2f}'\
                     f'_rfac{args.rfac:.2f}_rtrunc{int(args.r_trunc)}'))
        np.save(opj(args.odir, f'radii_rfac{args.rfac:.2f}_rtrunc{int(args.r_trunc)}'),
                radii)
