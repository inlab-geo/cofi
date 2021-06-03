! A common block to hold variables that we want to be able to 
! initialize here and then use in our forward modelling.
module globvars
        integer :: npt=7
        real :: voro(7,3) ! reference model. Initialized in cofi_init 
        integer :: mtype=0
        real :: fs  ! initialized in cofi_init
        real :: gauss_a ! initialized in cofi_init 
        real :: water_c ! initialized in cofi_init 
        real :: angle=35.0
        real :: time_shift=5.0
        real :: v60=8.043
        integer :: ndatar=626
        real :: time_obs(626)
        real :: val_obs(626)
end module


! CoFI interface code. This is called once when the RFC code is loaded by CoFI
subroutine cofi_init(fs0, gauss_a0, water_c0, true_model)
    !F2PY INTENT(IN) :: fs
    !F2PY INTENT(IN) :: gauss_a
    !F2PY INTENT(IN) :: water_c
    !F2PY INTENT(IN) :: true_model
    !F2PY real  :: fs
    !F2PY real :: gauss_a
    !F2PY real :: water_c
    !F2PY real :: true_model(7,3)

    use globvars
    implicit none
    real fs0, gauss_a0, water_c0
    real :: true_model(7,3)

    ! initialize common block variables with values passed in
    voro = true_model
    fs = fs0
    gauss_a = gauss_a0
    water_c = water_c0

    ! generate the receiver function using the reference model, and store the results in our globvars module
    ! We'll need this later to calculate misfit
    call RFcalc_nonoise(voro,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,npt,time_obs,val_obs)

end


! CoFI interface code for calculating misfit of a model
! For convenience, it also returns:
!     * the signal produced by model 
!     * the signal produced by the reference model
subroutine cofi_misfit(model,misfit,value_pred,value_obs)
    !F2PY INTENT(IN) :: model
    !F2PY INTENT(OUT) :: misfit
    !F2PY INTENT(OUT) :: value_pred
    !F2PY INTENT(OUT) :: value_obs
    !F2PY REAL :: model(7,3)
    !F2PY REAL :: value_pred(626)
    !F2PY REAL :: value_obs(626)
    !F2PY REAL :: misfit
    ! Calculate misfit against the reference model
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

