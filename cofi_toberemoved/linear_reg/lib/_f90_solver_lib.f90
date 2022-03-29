! FILE: _f90_solver_lib.f90

      module f90_lr_mod
      contains

! ****************** SUBROUTINE: SOLVE ***********************
        subroutine solve(m, n, g, y, res)

!f2py intent(in) m
!f2py intent(in) n
!f2py intent(in) g
!f2py intent(in) y
!f2py intent(out) res
!f2py depend(m,n) g
!f2py depend(n) y
!f2py depend(m) res

        integer :: m, n
        double precision :: g(n,m), y(n), res(m)
        double precision :: gtg(m,m), gtg_inv(m,m), gtg_inv_gt(m,n)

        gtg = matmul(transpose(g), g)
        call inverse(m, gtg, gtg_inv)
        gtg_inv_gt = matmul(gtg_inv, transpose(g))
        res = reshape(matmul(gtg_inv_gt, reshape(y, [n,1])), [m])

        end subroutine


! **************** SUBROUTINE: MATRIX INVERSE ************************
        subroutine inverse(n, mat, res)
!f2py intent(in) n
!f2py intent(in) mat
!f2py intent(out) res
!f2py depend(n) mat
!f2py depend(n) res
        implicit none
        integer :: n
        integer :: p, q, a, b, i, j
        double precision :: mat(n,n), res(n,n)
        double precision :: det, det_tmp, tmp(n,n), fac(n,n)
        call get_determinant(n, mat, det)
        do q=1,n
          do p=1,n
            a = 0
            b = 0
            call get_cofactor(n, q, p, mat, tmp)
            call get_determinant(n-1, tmp, det_tmp)
            fac(q,p) = (-1)**(q+p) * det_tmp
          end do
        end do
        do i=1,n
          do j=1,n
            res(i,j) = fac(j,i) / det
          end do
        end do
        end subroutine

! !alculating determinant of a matrix
! ref: http://web.hku.hk/~gdli/UsefulFiles/Example-Fortran-program.html
        subroutine get_determinant(n, mat, det)
!f2py intent(in) n
!f2py intent(in) mat
!f2py intent(out) det
!f2py depend(n) mat
        implicit none
        integer :: n, i, j, k
        double precision :: mat(n,n), mat_copy(n,n), m, tmp, l, det
        logical :: det_exists
        mat_copy = mat
        det_exists = .TRUE.
        l = 1.0
        do k=1,n-1
          if (mat_copy(k,k) == 0) then
            det_exists = .FALSE.
            do i=k+1,n
              if (mat_copy(i,k) /= 0) then
                do j=1,n
                  tmp = mat_copy(i,j)
                  mat_copy(i,j) = mat_copy(k,j)
                  mat_copy(k,j) = tmp
                end do
                det_exists = .TRUE.
                l = -l
                exit
              endif
            end do
            if (det_exists .EQV. .FALSE.) then
              det = 0.0
              return
            endif
          endif
          do j=k+1,n
            m = mat_copy(j,k) / mat_copy(k,k)
            do i=k+1,n
              mat_copy(j,i) = mat_copy(j,i) - m*mat_copy(k,i)
            end do
          end do
        end do
        det = l
        do i=1,n
          det = det * mat_copy(i,i)
        end do
        return
        end subroutine

        subroutine get_cofactor(n, p, q, mat, res)
!f2py intent(in) n
!f2py intent(in) p
!f2py intent(in) q
!f2py intent(in) mat
!f2py intent(out) res
!f2py depend(n) mat
!f2py depend(n) res
        integer :: n, p, q, i, j, row, col
        double precision :: mat(n,n), res(n-1,n-1)
        i = 1
        j = 1
        do row=1,n
          do col=1,n
            if (row /= p .AND. col /= q) then
              res(i,j) = mat(row,col)
              j = j + 1
              if (j .EQ. (n-1)) then
                j = 1
                i = i + 1
              endif
            endif
          end do
        end do
        end subroutine

! **************** SUBROUTINE: DISPLAY MATRIX ************************
        subroutine display(m, n, mat)
!f2py intent(in) m
!f2py intent(in) n
!f2py intent(in) mat
!f2py depend(m,n) mat
        integer :: m, n, i
        double precision :: mat(m,n)
        do i=1,m
          write(*,'(999f8.3)') mat(i,:)
        end do
        write(*,*)
        end subroutine

! **************** SUBROUTINE: DISPLAY VE!TOR ************************
        subroutine display_vec(n, vec)
!f2py intent(in) n
!f2py intent(in) vec
!f2py depend(n) vec
        integer :: n
        double precision :: vec(n)
        write(*,'(999f8.3)') vec
        write(*,*)
        end subroutine

        subroutine hello()
        write ( *, '(a)' ) '  Hello, world!'
        end subroutine

      end module f90_lr_mod

! *********************** MAIN ***************************************
      program main
      use f90_lr_mod
      integer :: m = 2, n = 3
      double precision :: g(3,2), y(3), res(2)
      g = reshape((/ 0, 1, 1, 1, 0, 2 /), shape(g))
      data y/ 6, 12, 6 /
      call solve(m, n, g, y, res)
      call display_vec(m, res)
      stop
      end
! END FILE _f90_solver_lib.f90