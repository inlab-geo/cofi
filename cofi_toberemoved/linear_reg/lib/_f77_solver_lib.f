C FILE: _f77_solver_lib.f

C ****************** SUBROUTINE: SOLVE ***********************
      subroutine solve(m, n, g, y, res)

Cf2py intent(in) m
Cf2py intent(in) n
Cf2py intent(in) g
Cf2py intent(in) y
Cf2py intent(out) res
Cf2py depend(m,n) g
Cf2py depend(n) y
Cf2py depend(m) res

      integer m, n
      double precision g(n,m), y(n), res(m)
      double precision gtg(m,m), gtg_inv(m,m), gtg_inv_gt(m,n)

      call matrix_mult_transA(n, m, m, g, g, gtg)
      call inverse(m, gtg, gtg_inv)
      call matrix_mult_transB(m, m, n, gtg_inv, g, gtg_inv_gt)
      call matrix_mult_vect(m, n, gtg_inv_gt, y, res)

      end subroutine solve

C **************** SUBROUTINE: MATRIX MULT (TRANS A) ****************
      subroutine matrix_mult_transA(m, n, k, a, b, res)
Cf2py intent(in) m
Cf2py intent(in) n
Cf2py intent(in) k
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(out) res
Cf2py depend(m,n) a
Cf2py depend(m,k) b
Cf2py depend(n,k) res
      integer m, n, k, i, j, p
      double precision a(m,n), b(m,k), res(n,k)
      do 13 i=1,n
        do 12 j=1,k
          res(i,j) = 0
          do 11 p=1,m
            res(i,j) = res(i,j) + a(p,i) * b(p,j)
11        continue
12      continue
13    continue
      end subroutine matrix_mult_transA

C **************** SUBROUTINE: MATRIX MULT (TRANS B) ****************
      subroutine matrix_mult_transB(m, n, k, a, b, res)
Cf2py intent(in) m
Cf2py intent(in) n
Cf2py intent(in) k
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(out) res
Cf2py depend(m,n) a
Cf2py depend(k,n) b
Cf2py depend(m,k) res
      integer m, n, k, i, j, p
      double precision a(m,n), b(k,n), res(m,k)
      do 16 i=1,m
        do 15 j=1,k
          res(i,j) = 0
          do 14 p=1,n
            res(i,j) = res(i,j) + a(i,p) * b(j,p)
14        continue
15      continue
16    continue
      end subroutine matrix_mult_transB

C **************** SUBROUTINE: MATRIX MULT VECTOR *******************
      subroutine matrix_mult_vect(m, n, a, b, res)
Cf2py intent(in) m
Cf2py intent(in) n
Cf2py intent(in) a
Cf2py intent(in) b
Cf2py intent(out) res
Cf2py depend(m,n) a
Cf2py depend(n) b
Cf2py depend(m) res
      integer m, n, i, j
      double precision a(m,n), b(n), res(m)
      do 18 i=1,m
        res(i) = 0
        do 17 j=1,n
          res(i) = res(i) + a(i,j) * b(j)
17      continue
18    continue
      end subroutine matrix_mult_vect

C **************** SUBROUTINE: MATRIX INVERSE ************************
      subroutine inverse(n, mat, res)
Cf2py intent(in) n
Cf2py intent(in) mat
Cf2py intent(out) res
Cf2py depend(n) mat
Cf2py depend(n) res
      implicit none
      integer n
      integer p, q, a, b, i, j
      double precision mat(n,n), res(n,n)
      double precision det, det_tmp, tmp(n,n), fac(n,n)
      call get_determinant(n, mat, det)
      do 20 q=1,n
        do 19 p=1,n
          a = 0
          b = 0
          call get_cofactor(n, q, p, mat, tmp)
          call get_determinant(n-1, tmp, det_tmp)
          fac(q,p) = (-1)**(q+p) * det_tmp
19      continue
20    continue
      do 22 i=1,n
        do 21 j=1,n
          res(i,j) = fac(j,i) / det
21      continue
22    continue
      end subroutine inverse

C Calculating determinant of a matrix
C ref: http://web.hku.hk/~gdli/UsefulFiles/Example-Fortran-program.html
      subroutine get_determinant(n, mat, det)
Cf2py intent(in) n
Cf2py intent(in) mat
Cf2py intent(out) det
Cf2py depend(n) mat
      implicit none
      integer n, i, j, k
      double precision mat(n,n), mat_copy(n,n), m, tmp, l, det
      logical det_exists
      intent(in) n
      intent(in) mat
      mat_copy = mat
      det_exists = .TRUE.
      l = 1.0
      do 27 k=1,n-1
        if (mat_copy(k,k) == 0) then
          det_exists = .FALSE.
          do 24 i=k+1,n
            if (mat_copy(i,k) /= 0) then
              do 23 j=1,n
                tmp = mat_copy(i,j)
                mat_copy(i,j) = mat_copy(k,j)
                mat_copy(k,j) = tmp
23            continue
              det_exists = .TRUE.
              l = -l
              exit
            endif
24        continue
          if (det_exists .EQV. .FALSE.) then
            det = 0.0
            return
          endif
        endif
        do 26 j=k+1,n
          m = mat_copy(j,k) / mat_copy(k,k)
          do 25 i=k+1,n
            mat_copy(j,i) = mat_copy(j,i) - m*mat_copy(k,i)
25        continue
26      continue
27    continue
      det = l
      do 28 i=1,n
        det = det * mat_copy(i,i)
28    continue
      return
      end subroutine get_determinant

      subroutine get_cofactor(n, p, q, mat, res)
Cf2py intent(in) n
Cf2py intent(in) p
Cf2py intent(in) q
Cf2py intent(in) mat
Cf2py intent(out) res
Cf2py depend(n) mat
Cf2py depend(n) res
      integer n, p, q, i, j, row, col
      double precision mat(n,n), res(n-1,n-1)
      i = 1
      j = 1
      do 30 row=1,n
        do 29 col=1,n
          if (row /= p .AND. col /= q) then
            res(i,j) = mat(row,col)
            j = j + 1
            if (j .EQ. (n-1)) then
              j = 1
              i = i + 1
            endif
          endif
29      continue
30    continue
      end subroutine get_cofactor

C **************** SUBROUTINE: DISPLAY MATRIX ************************
      subroutine display(m, n, mat)
Cf2py intent(in) m
Cf2py intent(in) n
Cf2py intent(in) mat
Cf2py depend(m,n) mat
      integer m, n, i
      double precision mat(m,n)
      do 33 i=1,m
        write(*,'(999f8.3)') mat(i,:)
33    continue
      write(*,*)
      end subroutine display

C **************** SUBROUTINE: DISPLAY VECTOR ************************
      subroutine display_vec(n, vec)
Cf2py intent(in) n
Cf2py intent(in) vec
Cf2py depend(n) vec
      integer n
      double precision vec(n)
      write(*,'(999f8.3)') vec
      write(*,*)
      end subroutine display_vec

      subroutine hello()
      write ( *, '(a)' ) '  Hello, world!'
      end subroutine hello

C *********************** MAIN ***************************************
      program main
      integer m, n
      double precision g(3,2), y(3), res(2)
      m = 2
      n = 3
      g = reshape((/ 0, 1, 1, 1, 0, 2 /), shape(g))
      data y/ 6, 12, 6 /
      call solve(m, n, g, y, res)
      call display_vec(m, res)
      stop
      end
C END FILE _f77_solver_lib.f