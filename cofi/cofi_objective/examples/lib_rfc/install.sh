#! /usr/bin/env bash
cd RFsubs
/opt/anaconda3/bin/gfortran -c -O3 *.f*
cd ..
/opt/anaconda3/bin/f2py -m rfc -c RF.F90 RFsubs/*.f*
