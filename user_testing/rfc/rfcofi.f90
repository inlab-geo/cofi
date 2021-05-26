module globvars
        integer :: npt=7
        integer :: npt2=3
        real :: voro(7,3) ! reference model
        integer :: mtype=0
        real :: fs=25.0
        real :: gauss_a=2.5
        real :: water_c=0.0001
        real :: angle=35.0
        real :: time_shift=5.0
        real :: v60=8.043
        integer :: ndatar=626
        real :: time_obs(626)
        real :: val_obs(626)
end module


subroutine cofi_init()
use globvars
implicit none
! these values are for the reference model (in this case, it is the 'correct' 
! model).
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

! generate the receiver function using the reference model, and store the results in our globvars module
! We'll need this later to calculate misfit
call RFcalc_nonoise(voro,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,npt,time_obs,val_obs)

end


subroutine cofi_misfit(model,value_pred,value_obs,misfit)
!F2PY INTENT(IN) :: model
!F2PY INTENT(OUT) :: misfit
!F2PY INTENT(OUT) :: value_pred
!F2PY INTENT(OUT) :: value_obs
!F2PY REAL :: model(7,3)
!F2PY REAL :: value_pred(626)
!F2PY REAL :: value_obs(626)
!F2PY REAL :: misfit
! Calculate misfit for reference model
use globvars
real :: model(7,3) 
real :: value_pred(626) ! result of forward model for 'model'
real :: value_obs(626)  ! result of forward model for reference model
real :: tim2(626)       ! not used
real :: res(626)        ! residuals between predicted and observed
real :: misfit          ! Error between predicted and actual. Here we just use sum of squares for simplicity.

call RFcalc_nonoise(model,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,npt,tim2,value_pred)
res = value_pred-val_obs
value_obs = val_obs
misfit = sum(res*res)
end
