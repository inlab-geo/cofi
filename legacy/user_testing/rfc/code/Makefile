clean:
	rm -f *.so *.o RFsubs/*.o *.a

rfc:
	gfortran -c RFsubs/*.f RFsubs/*.f90 RF.f90
	ar crv rfc.a *.o
