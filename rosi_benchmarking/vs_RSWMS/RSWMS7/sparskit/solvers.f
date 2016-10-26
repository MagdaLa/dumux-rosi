     program riters
c-----------------------------------------------------------------------
c test program for iters -- the basic iterative solvers
c
c     this program reads a Harwell/Boeing matrix from standard input
c     and solves the linear system with an artifical right-hand side
c     (the solution is a vector of (1,1,...,1)^T)
c-----------------------------------------------------------------------
c      implicit none
c      implicit real*8 (a-h,o-z)
      integer nmax, nzmax, maxits,lwk
      parameter (nmax=5000,nzmax=100000,maxits=60,lwk=nmax*40)
      integer ia(nmax),ja(nzmax),jau(nzmax),ju(nzmax),iw(nmax*3)
      integer ipar(16),i,lfil,nwk,nrow,ierr
      real*8  a(nzmax),sol(nmax),rhs(nmax),au(nzmax),wk(nmax*40)
      real*8  xran(nmax), fpar(16), tol
      character guesol*2, title*72, key*8, type*3
      external cg,bcg,dbcg,bcgstab,tfqmr,gmres,fgmres,dqgmres
      external cgnr, fom, runrc, ilut
c
c     set the parameters for the iterative solvers
c
      ipar(2) = 2
      ipar(3) = 1
      ipar(4) = lwk
      ipar(5) = 16
      ipar(6) = maxits
      fpar(1) = 1.0D-5
      fpar(2) = 1.0D-10
c--------------------------------------------------------------
c     read in a matrix from standard input
c--------------------------------------------------------------
      iounit = 5
      job = 2
      nrhs = 0
      call readmt (nmax,nzmax,job,iounit,a,ja,ia,a,nrhs,
     *     guesol,nrow,ncol,nnz,title,key,type,ierr)
      print *, 'READ the matrix ', key, type
      print *, title
      print *
c
c     set-up the preconditioner ILUT(15, 1E-4) ! new definition of lfil
c
      lfil = 15
      tol = 1.0D-4 ! this is too high for ilut for saylr1
      tol = 1.0D-7
      nwk = nzmax
      call ilut (nrow,a,ja,ia,lfil,tol,au,jau,ju,nwk,
     *     wk,iw,ierr)
      ipar(2) = 2
c
c     generate a linear system with known solution
c
      do i = 1, nrow
         sol(i) = 1.0D0
         xran(i) = 0.D0
      end do
      call amux(nrow, sol, rhs, a, ja, ia)
      print *, ' '
      print *, '	*** CG ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     cg)
      print *, ' '
      print *, '	*** BCG ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     bcg)
      print *, ' '
      print *, '	*** DBCG ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     dbcg)
      print *, ' '
      print *, '	*** CGNR ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     cgnr)
      print *, ' '
      print *, '	*** BCGSTAB ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     bcgstab)
      print *, ' '
      print *, '	*** TFQMR ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     tfqmr)
      print *, ' '
      print *, '	*** FOM ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     fom)
      print *, ' '
      print *, '	*** GMRES ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     gmres)
      print *, ' '
      print *, '	*** FGMRES ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     fgmres)
      print *, ' '
      print *, '	*** DQGMRES ***'
      call runrc(nrow,rhs,sol,ipar,fpar,wk,xran,a,ja,ia,au,jau,ju,
     +     dqgmres)
      stop
      end
c-----end-of-main
c-----------------------------------------------------------------------

   subroutine runrc(n,rhs,sol,ipar,fpar,wk,guess,a,ja,ia,
     +     au,jau,ju,solver)
      implicit none
      integer n,ipar(16),ia(n+1),ja(*),ju(*),jau(*)
      real*8 fpar(16),rhs(n),sol(n),guess(n),wk(*),a(*),au(*)
      external solver
c-----------------------------------------------------------------------
c     the actual tester. It starts the iterative linear system solvers
c     with a initial guess suppied by the user.
c
c     The structure {au, jau, ju} is assumed to have the output from
c     the ILU* routines in ilut.f.
c
c-----------------------------------------------------------------------
c     local variables
c
      integer i, iou, its
      real*8 res, dnrm2
c     real dtime, dt(2), time
c     external dtime
      external dnrm2
      save its,res
c
c     ipar(2) can be 0, 1, 2, please don't use 3
c
      if (ipar(2).gt.2) then
         print *, 'I can not do both left and right preconditioning.'
         return
      endif
c
c     normal execution
c
      its = 0
      res = 0.0D0
c
      do i = 1, n
         sol(i) = guess(i)
      enddo
c
      iou = 6
      ipar(1) = 0
c     time = dtime(dt)
 10   call solver(n,rhs,sol,ipar,fpar,wk)
c
c     output the residuals
c
      if (ipar(7).ne.its) then
         write (iou, *) its, real(res)
         its = ipar(7)
      endif
      res = fpar(5)
c
      if (ipar(1).eq.1) then
         call amux(n, wk(ipar(8)), wk(ipar(9)), a, ja, ia)
         goto 10
      else if (ipar(1).eq.2) then
         call atmux(n, wk(ipar(8)), wk(ipar(9)), a, ja, ia)
         goto 10
      else if (ipar(1).eq.3 .or. ipar(1).eq.5) then
         call lusol(n,wk(ipar(8)),wk(ipar(9)),au,jau,ju)
         goto 10
      else if (ipar(1).eq.4 .or. ipar(1).eq.6) then
         call lutsol(n,wk(ipar(8)),wk(ipar(9)),au,jau,ju)
         goto 10
      else if (ipar(1).le.0) then
         if (ipar(1).eq.0) then
            print *, 'Iterative sovler has satisfied convergence test.'
         else if (ipar(1).eq.-1) then
            print *, 'Iterative solver has iterated too many times.'
         else if (ipar(1).eq.-2) then
            print *, 'Iterative solver was not given enough work space.'
            print *, 'The work space should at least have ', ipar(4),
     &           ' elements.'
         else if (ipar(1).eq.-3) then
            print *, 'Iterative sovler is facing a break-down.'
         else
            print *, 'Iterative solver terminated. code =', ipar(1)
         endif
      endif
c     time = dtime(dt)
      write (iou, *) ipar(7), real(fpar(6))
      write (iou, *) '# retrun code =', ipar(1),
     +     '	convergence rate =', fpar(7)
c     write (iou, *) '# total execution time (sec)', time
c
c     check the error
c
      call amux(n,sol,wk,a,ja,ia)
      do i = 1, n
         wk(n+i) = sol(i) -1.0D0
         wk(i) = wk(i) - rhs(i)
      enddo
      write (iou, *) '# the actual residual norm is', dnrm2(n,wk,1)
      write (iou, *) '# the error norm is', dnrm2(n,wk(1+n),1)
c
      if (iou.ne.6) close(iou)
      return
      end
c-----end-of-runrc
c-----------------------------------------------------------------------
      function distdot(n,x,ix,y,iy)
      integer n, ix, iy
      real*8 distdot, x(*), y(*), ddot
      external ddot
      distdot = ddot(n,x,ix,y,iy)
      return
      end
c-----end-of-distdot
c-----------------------------------------------------------------------
c
      function afun (x,y,z)
      real*8 afun, x,y, z 
      afun = -1.0D0
      return 
      end
      
      function bfun (x,y,z)
      real*8 bfun, x,y, z 
      bfun = -1.0D0
      return 
      end
      
      function cfun (x,y,z)
      real*8 cfun, x,y, z 
      cfun = -1.0D0
      return 
      end
      
      function dfun (x,y,z)
      real*8 dfun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      dfun = gammax*exp(x*y)
      return 
      end
      
      function efun (x,y,z)
      real*8 efun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      efun = gammay*exp(-x*y) 
      return 
      end
      
      function ffun (x,y,z)
      real*8 ffun, x,y, z 
      ffun = 0.0D0
      return 
      end
      
      function gfun (x,y,z)
      real*8 gfun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      gfun = alpha 
      return 
      end
      
      function hfun (x,y,z)
      real*8 hfun, x,y, z, gammax, gammay, alpha
      common /func/ gammax, gammay, alpha
      hfun = alpha * sin(gammax*x+gammay*y-z)
      return 
      end
      

      function betfun(side, x, y, z)
      real*8 betfun, x, y, z
      character*2 side
      betfun = 1.0
      return
      end

      function gamfun(side, x, y, z)
      real*8 gamfun, x, y, z
      character*2 side
      if (side.eq.'x2') then
         gamfun = 5.0
      else if (side.eq.'y1') then
         gamfun = 2.0
      else if (side.eq.'y2') then
         gamfun = 7.0
      else
         gamfun = 0.0
      endif
      return
      end
c-----------------------------------------------------------------------
c     functions for the block PDE's 
c-----------------------------------------------------------------------
      subroutine afunbl (nfree,x,y,z,coeff)
      return
      end
c     
      subroutine bfunbl (nfree,x,y,z,coeff)
      return 
      end
      
      subroutine cfunbl (nfree,x,y,z,coeff)
c     
      return 
      end
      
      subroutine dfunbl (nfree,x,y,z,coeff)
      
      return
      end
c     
      subroutine efunbl (nfree,x,y,z,coeff)
      return 
      end
c     
      subroutine ffunbl (nfree,x,y,z,coeff)
      return 
      end
c     
      subroutine gfunbl (nfree,x,y,z,coeff)
      return 
      end
      
