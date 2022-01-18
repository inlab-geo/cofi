subroutine whichcell(point,voro,npt,npt_max,ind)
real point,voro(npt_max,3)
integer npt,ind,i,npt_max

ind=1
do i=1,npt
!write(*,*)point,abs(point-voro(i,1)),abs(point-voro(ind,1))
    if (abs(point-voro(i,1))<abs(point-voro(ind,1))) then
        ind=i
    endif
 !   write(*,*)ind
enddo
return
end