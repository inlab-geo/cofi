#! /usr/bin/env bash
cd RFsubs
gfortran -c -O3 *.f*
cd ..
/Users/malcolm/Library/Enthought/Canopy/edm/envs/User/bin/f2py -m rfc -c RF.F90 RFsubs/*.f*
