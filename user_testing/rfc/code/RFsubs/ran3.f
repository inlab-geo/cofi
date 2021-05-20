c-------------------------------------------------------------------
c
c     Numerical Recipes random number generator
c
c ----------------------------------------------------------------------------

      FUNCTION ran3(idum)
      INTEGER idum
      INTEGER MBIG,MSEED,MZ
c     REAL MBIG,MSEED,MZ
      REAL ran3,FAC
      PARAMETER (MBIG=1000000000,MSEED=161803398,MZ=0,FAC=1./MBIG)
c     PARAMETER (MBIG=4000000.,MSEED=1618033.,MZ=0.,FAC=1./MBIG)
      INTEGER i,iff,ii,inext,inextp,k
      INTEGER mj,mk,ma(55)
c     REAL mj,mk,ma(55)
      SAVE iff,inext,inextp,ma
      DATA iff /0/
c     write(*,*)' idum ',idum
      if(idum.lt.0.or.iff.eq.0)then
         iff=1
         mj=MSEED-iabs(idum)
         mj=mod(mj,MBIG)
         ma(55)=mj
         mk=1
         do 11 i=1,54
            ii=mod(21*i,55)
            ma(ii)=mk
            mk=mj-mk
            if(mk.lt.MZ)mk=mk+MBIG
            mj=ma(ii)
c           write(*,*)' idum av',idum
 11      continue
         do 13 k=1,4
         do 12 i=1,55
            ma(i)=ma(i)-ma(1+mod(i+30,55))
            if(ma(i).lt.MZ)ma(i)=ma(i)+MBIG
 12      continue
 13      continue
c        write(*,*)' idum ap',idum
         inext=0
         inextp=31
         idum=1
      endif
c     write(*,*)' idum app ',idum
      inext=inext+1
      if(inext.eq.56)inext=1
      inextp=inextp+1
      if(inextp.eq.56)inextp=1
      mj=ma(inext)-ma(inextp)
      if(mj.lt.MZ)mj=mj+MBIG
      ma(inext)=mj
      ran3=mj*FAC
c     write(*,*)' idum ',idum
      return
      END
