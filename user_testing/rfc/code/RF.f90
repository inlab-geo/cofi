Subroutine RFcalc_nonoise(voro,mtype,fs,gauss_a,water_c,angle,time_shift,ndatar,v60,npt,time,wdata) 
!F2PY INTENT(OUT) :: wdata
!F2PY INTENT(OUT) :: time
!F2PY INTENT(IN) :: voro
!F2PY INTENT(IN) :: mtype
!F2PY INTENT(IN) :: fs
!F2PY INTENT(IN) :: gauss_a
!F2PY INTENT(IN) :: water_c
!F2PY INTENT(IN) :: angle
!F2PY INTENT(IN) :: time_shift
!F2PY INTENT(IN) :: ndatar
!F2PY INTENT(IN) :: v60
!F2PY INTENT(HIDE),DEPEND(voro) :: npt
!F2PY REAL :: voro(npt,3)
implicit none

!------------------------------------------------
! Prior distribution (Bounds odf the model space)
!------------------------------------------------

real, parameter :: d_min = 0
real, parameter :: d_max = 60
      

!--------------------------------------------
! Parameters for Receiver Function
!-------------------------------------------- 

! All parameter converted to input variables 27/06/18

 real fs,gauss_a,water_c,angle,time_shift,v60 
 integer ndatar,mtype
! Parameters are associated with the data vector which is a waveform

! real, parameter :: fs = 25.00       !sampling frequency
! real, parameter :: gauss_a = 2.5    !number 'a' defining the width of 
                                      !the gaussian filter in the deconvolution  
! real, parameter :: water_c = 0.0001 !water level for deconvolution 
! real, parameter :: angle = 35       ! angle of the incoming wave 
! real, parameter :: time_shift = 5   !time shift before the first p pusle 
! integer, parameter :: ndatar = 157  !Number of data points
! integer, parameter :: ndatar = 626  !Number of data points
! real, parameter ::    v60 = 8.043      ! Vs at 60km depth 
! mtype is the indicator for the format of the velocity
!v60 is needed to compute the ray parameter from angle. 

 real, parameter ::    pi = 3.14159265      !don't touch
 real, parameter ::    rad = 0.017453292    !don't touch

!%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

!***************************************************************

! DECLARATIONS OF VARIABLES

!****************************************************************

real wdata(ndatar),time(ndatar)
real voro(npt,3)
real din,ppara,depth
integer i,npt
real beta(npt),h(npt),vpvs(npt),qa(npt),qb(npt)

!write(*,*)' npt',npt
!write(*,*)' ndatar',ndatar

!**************************************************************

!    Set up reference model 

!**************************************************************

if(mtype.eq.0)then
     call voro2qmodel(voro,npt,npt,d_min,d_max,beta,h,vpvs,qa,qb)
else if(mtype.eq.1)then
     do i=1,npt
        h(i)= voro(i,1) 
        beta(i) = voro(i,2)
        vpvs(i) = voro(i,3)
        qa(i)= 1450
        qb(i)= 600
     enddo
     h(npt)=0
else if(mtype.eq.2)then
     depth = 0.0
     do i=1,npt
        h(i)= voro(i,1) - depth 
        beta(i) = voro(i,2)
        vpvs(i) = voro(i,3)
        qa(i)= 1450
        qb(i)= 600
        depth = voro(i,1)
     enddo
     h(npt)=0
else 
     write(*,*)' Fortran routine RF: Input model type unknown'
     stop
end if

! Compute RF for current model
ppara=sin(angle*rad)/v60
din=asin(ppara*beta(npt)*vpvs(npt))/rad

call theo(&
          npt, beta, h, vpvs, qa, qb, fs, din,&
          gauss_a, water_c, time_shift, ndatar, wdata )

do i=1,ndatar
   time(i) = -time_shift+(i-1)/fs
enddo
return
end


