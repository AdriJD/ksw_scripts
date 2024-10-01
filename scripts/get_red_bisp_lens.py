'''
Compute the reduced bispectra due to the ISW-lensing correlation.
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
    args = parser.parse_args()

    os.makedirs(args.odir, exist_ok=True)
    
    cosmo_opts = dict(H0=67.74, ombh2=0.02230, omch2=0.1188,
                      mnu=0.06, omk=0, tau=0.066, TCMB=2.7255)

    pars = camb.CAMBparams()
    pars.set_cosmology(**cosmo_opts)
    ip = camb.initialpower.InitialPowerLaw()
    ip.set_params(As=2.10058e-9, ns=0.96, pivot_scalar=0.05)
    pars.set_initial_power(ip)
    cosmo = Cosmology(pars)

    cosmo.compute_transfer(args.lmax + 300)
    cosmo.compute_c_ell(lmax=args.lmax)
    cosmo.write_transfer(opj(args.odir, f'transfer_lmax{args.lmax}'))
    cosmo.write_c_ell(opj(args.odir, f'c_ell_lmax{args.lmax}'))

    cosmo.add_ttt_lensing_bispectrum()
    rb = cosmo.red_bispectra[0]
    rb.write(opj(args.odir, f'isw_lens_lmax{args.lmax}'))
