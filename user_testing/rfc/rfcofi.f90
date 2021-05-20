
subroutine cofi_init(mtype, fs, gauss_a, water_c, angle, time_shift, ndatar, v60, time_obs, value_obs)
!F2PY INTENT(OUT) :: mtype
!F2PY INTENT(OUT) :: fs
!F2PY INTENT(OUT) :: gauss_a
!F2PY INTENT(OUT) :: water_c
!F2PY INTENT(OUT) :: angle
!F2PY INTENT(OUT) :: time_shift
!F2PY INTENT(OUT) :: ndatar
!F2PY INTENT(OUT) :: v60
!F2PY INTENT(OUT) :: time_obs
!F2PY INTENT(OUT) :: value_obs
real voro(7,3)
integer ndatar,mtype,npt
real time_obs(626)
real value_obs(626)
real fs, gauss_a, water_c, angle, time_shift, v60

mtype=0
fs=25.0
gauss_a=2.5
water_c=0.0001
angle=35.0
time_shift=5.0
ndatar=626
v60=8.043
npt=7

voro(1,1) = 8.370596
voro(1,2) = 3.249075
voro(1,3) = 1.7
voro(2,1) = 17.23163
voro(2,2) = 3.001270
voro(2,3) = 1.7
voro(3,1) = 1.9126695E-02
voro(3,2) = 2.509443
voro(3,3) = 1.7
voro(4,1) = 19.78145
voro(4,2) = 3.562691
voro(4,3) = 1.7
voro(5,1) = 41.73066
voro(5,2) = 4.225965
voro(5,3) = 1.7
voro(6,1) = 14.35261
voro(6,2) = 2.963322
voro(6,3) = 1.7
voro(7,1) = 49.92358
voro(7,2) = 4.586726
voro(7,3) = 1.7

call RFcalc_nonoise(voro,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,npt,time_obs,value_obs)

end

subroutine fwd(model,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,time,val)
!F2PY INTENT(OUT) :: time
!F2PY INTENT(OUT) :: val
!F2PY INTENT(IN) :: model
!F2PY INTENT(IN) :: mtype
!F2PY INTENT(IN) :: fs
!F2PY INTENT(IN) :: gauss_a
!F2PY INTENT(IN) :: water_c
!F2PY INTENT(IN) :: angle
!F2PY INTENT(IN) :: time_shift
!F2PY INTENT(IN) :: ndatar
!F2PY INTENT(IN) :: v60
!F2PY REAL :: model(7,3)
real val(ndatar)
real time(ndatar)
integer ndatar,mtype
real fs, gauss_a, water_c, angle, time_shift, v60 
real model(7,3)

call RFcalc_nonoise(model,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,time,val)

end

subroutine cofi_misfit(model,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,value_obs,misfit)
!F2PY INTENT(OUT) :: misfit
!F2PY INTENT(OUT) :: value_obs
!F2PY INTENT(IN) :: model
!F2PY INTENT(IN) :: gauss_a
!F2PY INTENT(IN) :: water_c
!F2PY INTENT(IN) :: angle
!F2PY INTENT(IN) :: time_shift
!F2PY INTENT(IN) :: ndatar
!F2PY INTENT(IN) :: v60
!F2PY REAL :: model(7,3)
! Calculate waveform misfit for reference model
real value_obs(ndatar)
real tim2(ndatar)
real RFp(ndatar)
real res(ndatar)
integer ndatar,mtype
real fs, gauss_a, water_c, angle, time_shift, v60, misfit
real model(:,:)

call RFcalc_nonoise(model,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,tim2,RFp)
res = RFp-value_obs
misfit = sum(res*res)
end






