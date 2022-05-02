#! /usr/bin/env bash
cd RFsubs
gfortran -c -O3 *.f*
cd ..
f2py-2.7 -m rfc -c RF.F90 RFsubs/*.f*
