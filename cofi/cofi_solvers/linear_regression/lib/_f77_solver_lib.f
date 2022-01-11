C FILE: _f77_solver_lib.f
      subroutine fib(a, n)
      integer n
      real*8 a(n)
Cf2py intent(in) n
Cf2py intent(out) a
Cf2py depend(n) a
      do i=1,n
          if (i .EQ. 1) then
              a(i) = 0.0D0
          elseif (i .EQ. 2) then
              a(i) = 1.0D0
          else
              a(i) = a(i-1) + a(i-2)
          end if
      enddo
      end

      subroutine hello()
      write ( *, '(a)' ) '  Hello, world!'
      end

      program main
      call hello()
      stop
      end
C END FILE _f77_solver_lib.f