subroutine voro2qmodel(voro,npt,npt_max,d_min,d_max,beta,h,vpvs,qa,qb)

real voro(npt_max,3),d_min,d_max,beta(npt_max),vpvs(npt_max),qa(npt_max),qb(npt_max),maxx,minn,summ,h(npt_max) 
integer npt,ind,i,j,order(npt),npt_max

beta=0
vpvs=0
qa=0
qb=0
h=0

! do i=1,npt
! write(*,*)voro(i,:)
! enddo


ind=1


do i=1,npt
!write(*,*)'ordre',i
	maxx=d_max
	if (i==1) then
		minn=d_min
	else
		minn=voro(order(i-1),1)
	endif
!	write(*,*)minn,maxx
	do j=1,npt
		if ((minn<voro(j,1)).and.(voro(j,1)<maxx)) then
!			write(*,*)minn,voro(j,1),maxx
			ind=j
			maxx=voro(j,1)
!			write(*,*)ind,maxx
		endif
	enddo
	order(i)=ind
! 	write(*,*)ind
! 	write(*,*)
!   	write(*,*)

enddo


summ=0
do i=1,npt-1

h(i)= (voro(order(i),1)+voro(order(i+1),1))/2 - summ
summ = summ + h(i)
beta(i) = voro(order(i),2)
vpvs(i) = voro(order(i),3)

qa(i)= 1450
qb(i)= 600 

enddo

h(npt)=0
beta(npt) = voro(order(npt),2)
vpvs(npt) = voro(order(npt),3)

qa(npt)= 1450
qb(npt)= 600 

return
end
