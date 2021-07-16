!Manlin Chawla 01205586
!M3C 2018 Homework 3

!To compile: gfortran -fopenmp -O3 -o main.exe hw3_dev.f90 hw3_main.f90

!This module contains four module variables and two subroutines;
!one of these routines must be developed for this assignment.
!Module variables--
! tr_b, tr_e, tr_g: the parameters b, e, and g=gamma in the tribe competition model
! numthreads: The number of threads that should be used in parallel regions within simulate2_omp
!
!Module routines---
! simulate2_f90: Simulate tribal competition over m trials. Return: all s matrices at final time
! and fc at nt+1 times averaged across the m trials.
! simulate2_omp: Same input/output functionality as simulate2.f90 but parallelized with OpenMP

module tribes
  use omp_lib
  implicit none
  integer :: numthreads
  real(kind=8) :: tr_b,tr_e,tr_g
contains

!-------------------------------------------------------------------------------------------------------------------------------------------------------------
!Simulate m trials of Cooperator vs. Mercenary competition using the parameters, tr_b and tr_e.
!Input:
! n: defines n x n grid of villages
! nt: number of time steps
! m: number of trials
!Output:
! s: status matrix at final time step for all m trials
! fc_ave: fraction of cooperators at each time step (including initial condition)
! averaged across the m trials
subroutine simulate2_f90(n,nt,m,s,fc_ave)
  implicit none
  integer, intent(in) :: n,nt,m
  integer, intent(out), dimension(n,n,m) :: s
  real(kind=8), intent(out), dimension(nt+1) :: fc_ave
  integer :: i1,j1
  real(kind=8) :: n2inv
  integer, dimension(n,n,m) :: nb,nc
  integer, dimension(n+2,n+2,m) :: s2
  real(kind=8), dimension(n,n,m) :: f,p,a,pden,nbinv
  real(kind=8), dimension(n+2,n+2,m) :: f2,f2s2
  real(kind=8), allocatable, dimension(:,:,:) :: r !random numbers


  !---Problem setup----
  !Initialize arrays and problem parameters

  !initial condition
  s=1
  j1 = (n+1)/2
  s(j1,j1,:) = 0

  n2inv = 1.d0/dble(n*n)
  fc_ave(1) = sum(s)*(n2inv/m)

  s2 = 0
  f2 = 0.d0

  !Calculate number of neighbors for each point
  nb = 8
  nb(1,2:n-1,:) = 5
  nb(n,2:n-1,:) = 5
  nb(2:n-1,1,:) = 5
  nb(2:n-1,n,:) = 5
  nb(1,1,:) = 3
  nb(1,n,:) = 3
  nb(n,1,:) = 3
  nb(n,n,:) = 3

  nbinv = 1.d0/nb
  allocate(r(n,n,m))
  !---finished Problem setup---


  !----Time marching----
  do i1=1,nt

    call random_number(r) !Random numbers used to update s every time step

    !Set up coefficients for fitness calculation in matrix, a
    a = 1
    where(s==0)
      a=tr_b
    end where

    !create s2 by adding boundary of zeros to s
    s2(2:n+1,2:n+1,:) = s

    !Count number of C neighbors for each point
    nc = s2(1:n,1:n,:) + s2(1:n,2:n+1,:) + s2(1:n,3:n+2,:) + &
         s2(2:n+1,1:n,:)                  + s2(2:n+1,3:n+2,:) + &
         s2(3:n+2,1:n,:)   + s2(3:n+2,2:n+1,:)   + s2(3:n+2,3:n+2,:)

    !Calculate fitness matrix, f----
    f = nc*a
    where(s==0)
      f = f + (nb-nc)*tr_e
    end where
    f = f*nbinv
    !-----------

    !Calculate probability matrix, p----
    f2(2:n+1,2:n+1,:) = f
    f2s2 = f2*s2

    !Total fitness of cooperators in community
    p = f2s2(1:n,1:n,:) + f2s2(1:n,2:n+1,:) + f2s2(1:n,3:n+2,:) + &
           f2s2(2:n+1,1:n,:) + f2s2(2:n+1,2:n+1,:)  + f2s2(2:n+1,3:n+2,:) + &
          f2s2(3:n+2,1:n,:)   + f2s2(3:n+2,2:n+1,:)   + f2s2(3:n+2,3:n+2,:)

    !Total fitness of all members of community
    pden = f2(1:n,1:n,:) + f2(1:n,2:n+1,:) + f2(1:n,3:n+2,:) + &
           f2(2:n+1,1:n,:) + f2(2:n+1,2:n+1,:)  + f2(2:n+1,3:n+2,:) + &
          f2(3:n+2,1:n,:)   + f2(3:n+2,2:n+1,:)   + f2(3:n+2,3:n+2,:)


    p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g) !probability matrix
    !----------

    !Set new affiliations based on probability matrix and random numbers stored in R
    s = 0
    where (R<=p)
        s = 1
    end where

    fc_ave(i1+1) = sum(s)*(n2inv/m)

  end do

end subroutine simulate2_f90
!-------------------------------------------------------------------------------------------------------------------------------------------------------------
!Simulate m trials of Cooperator vs. Mercenary competition using the parameters, tr_b and tr_e.
!Same functionality as simulate2_f90, but parallelized with OpenMP
!Parallel regions should use numthreads threads.

!Input:
! n: defines n x n grid of villages
! nt: number of time steps
! m: number of trials
!Output:
! s: status matrix at final time step for all m trials
! fc_ave: fraction of cooperators at each time step (including initial condition)
! averaged across the m trials

subroutine simulate2_omp(n,nt,m,s,fc_ave)
  !To parallelize the subroutine simulate_f90, the approach taken was to distribute
  !the m trials of each simulation across the paralell region. This is because each
  !trial is independent from another trial meaning there is no data dependency so
  !iterating over the m trials was ideal to parallellize. The added advantage of this is
  !as m becomes large the parallel region adds an element of speed so it takes less time
  !for the code to run than it would otherwise without the paralellization. I have
  !also parallelized sections of the initial problem set up to speed up the process.

  !Simulate_f90 uses memory inefficiently as m increases. To avoid similar inefficiencies
  !I changed the dimensions of many of the variables such as nb,nc, f, p, a, pden and nbinv
  !to two dimensional arrays. These variables are calculated for one value of m
  !then used to compute the new affiliations and the fc_ave. As the value of m changes,
  !new values of nb, nc, f, p, a, pden and nbinv are calculated and replaced never
  !being stored in memory. This avoids the memory inefficiencies of simulate2_f90.

  !During the initial problem set up I have intialized the matrices f2 and s2 to be
  !zero and then used first private. This initializes each thread's value to the value
  !set before the paralell region. In the paralell region each thread computes it's
  !own values of fc_ave and these need to be collected when the paralell region ends.
  !To avoid the use of OMP critical which causes the process to slow down, I used
  !a reduction instead. Each thread computes its own fc_ave and the reduction tells
  !the compiler to sum the fc_ave into one global fc_ave at the end of the parallell loop.
  implicit none

  !Declare input/output variables
  integer, intent(in) :: n,nt,m
  integer, intent(out), dimension(n,n,m) :: s
  real(kind=8), intent(out), dimension(nt+1) :: fc_ave

  !Declare further variables as needed
  real(kind=8), allocatable, dimension(:,:) :: r !random numbers
  integer :: j1,i2,i3,i4,j4,i5,j5,i6,j6
  real(kind=8) :: n2inv
  integer, dimension(n,n) :: nb,nc
  real(kind=8), dimension(n,n) :: f,p,a,pden,nbinv
  integer, dimension(n+2,n+2) :: s2
  real(kind=8), dimension(n+2,n+2) :: f2,f2s2
  real(kind=8), dimension(nt+1) :: partial_sumfc

  !Initial condition and r allocation (does not need to be parallelized)
  s=1
  j1 = (n+1)/2
  s(j1,j1,:) = 0
  allocate(r(n,n))
  !-----------------------------------------------------------------------------

  !n2inv is the reciprocal of total number of villages in the grid (N*N)
  n2inv = 1.d0/dble(n*n)
  !Initial fraction of collaborators in s
  fc_ave(1) = sum(s)

  !s2 is matrix s with boundary of zeros around it, makes it easier to sum around
  !f2 is matrix f with boundary of zeros around it, makes it easier to sum around
  !$OMP parallel do collapse(2)
  do i4=1,n+2
    do j4=1,n+2
    s2(i4,j4)=0
    f2(i4,j4)=0.d0
    end do
  end do
  !$OMP end parallel do

  !Number of neighbours for inside points = 8
  !Number of neighbours for points on boundary = 5
  !Number of neighbours for corners = 3
  nb(1,1) = 3
  nb(1,n) = 3
  nb(n,1) = 3
  nb(n,n) = 3

  !$OMP parallel do collapse(2)
  do i5=2,n-1
    do j5=2,n-1
    nb(i5,j5) = 8
    nb(1,i5) = 5
    nb(n,i5) = 5
    nb(i5,1) = 5
    nb(i5,n) = 5
    end do
  end do
  !$OMP end parallel do

  !Reciprocal of the total number of neighbours
  !$OMP parallel do collapse(2)
  do i6=1,n
    do j6=1,n
    nbinv(i6,j6) = 1.d0/nb(i6,j6)
    end do
  end do
  !$OMP end parallel do

  !--------------Finished Problem Set Up------------------------

  !-----------------Time Marching-------------------------------
  !$OMP parallel do firstprivate(f2,s2) private(i3,nc,f,p,a,pden,f2s2,partial_sumfc) reduction(+:fc_ave)
  do i2=1,m

    do i3=1,nt
      call random_number(r) !Random numbers used to update s every time step

      !Set up coefficients for fitness score calculation in matrix a
      a = 1
      where(s(:,:,i2)==0)
        a=tr_b
      end where

      !Set inner square of s2 to be s
      s2(2:n+1,2:n+1) = s(:,:,i2)

      !Counts number of C neighbours for every village, done by overlapping section of matrix s2
      nc = s2(1:n,1:n) + s2(1:n,2:n+1) + s2(1:n,3:n+2) + &
           s2(2:n+1,1:n)                  + s2(2:n+1,3:n+2) + &
           s2(3:n+2,1:n)   + s2(3:n+2,2:n+1)   + s2(3:n+2,3:n+2)

      !Calculate fitness matrix f
      !C villages get 1pt for every C neighbour (a=1), 0pt for every M neighbour
      !M villages get bpt for every C neighbour (a=b), ept for every M neighbour
      f = nc*a
      where(s(:,:,i2)==0)
        f = f + (nb-nc)*tr_e
      end where
      f = f*nbinv

      !Set inner square of f2 to be f
      f2(2:n+1,2:n+1) = f

      !f contains fitness scores for C and M
      !f2s2 contains all of the fitness scores for just C
      f2s2 = f2*s2

      !Total fitness of collaborators in community
      p = f2s2(1:n,1:n) + f2s2(1:n,2:n+1) + f2s2(1:n,3:n+2) + &
           f2s2(2:n+1,1:n) + f2s2(2:n+1,2:n+1)  + f2s2(2:n+1,3:n+2) + &
          f2s2(3:n+2,1:n)   + f2s2(3:n+2,2:n+1)   + f2s2(3:n+2,3:n+2)

      !Total fitness of all members of coommunity
      pden = f2(1:n,1:n) + f2(1:n,2:n+1) + f2(1:n,3:n+2) + &
              f2(2:n+1,1:n) + f2(2:n+1,2:n+1)  + f2(2:n+1,3:n+2) + &
              f2(3:n+2,1:n)   + f2(3:n+2,2:n+1)   + f2(3:n+2,3:n+2)

      !Final probability matrix P_c
      !Final probability matrix P_m = 1-P_c
      p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g)

      !Set new affiliations based on probability matrix and random numbers stored in r
      !For each point if corresponding random probability is less than the P_c:
      !change to a C, if not then change to M
      s(:,:,i2) = 0
      where (r(:,:)<=p)
        s(:,:,i2) = 1
      end where

      !Calculating sum of collaborators to use for fc calculations
      partial_sumfc(i3+1)=sum(s(:,:,i2))

    end do

    !Cumulative sum of partial sums
    fc_ave = fc_ave + partial_sumfc

  end do
!$OMP end parallel do

!Divide by N*N*m to get final fractions of Collaborators (fc)
fc_ave=dble(fc_ave)*(n2inv/m)
deallocate(r)

end subroutine simulate2_omp
!-------------------------------------------------------------------------------------------------------------------------------------------------------------

!The subroutine simulate3_omp is a modified version of simulate2_omp and is used
!to generate an animation for Question 4. This subroutine only outputs the state
!matrix S at the intial state and after each year Nt. To modify this code I
!changed the dimenison of S to be a matrix of dimension n x n x nt+1. I have
!also removed the variables m and fc_ave and the associated lines of code as these
!are not needed. This subroutine is called on in the Python function visualize and
!the output s is sliced accordingly so each year is displayed as a frame, this forms
!the animation.
subroutine simulate3_omp(n,nt,s)
  implicit none

  !Declare input/output variables
  integer, intent(in) :: n,nt
  integer, intent(out), dimension(n,n,nt+1) :: s

  !Declare further variables as needed
  real(kind=8), allocatable, dimension(:,:) :: r !random numbers
  integer :: i1,j1,i2,i3,i4,j4,i5,j5,i6,j6
  real(kind=8) :: n2inv
  integer, dimension(n,n) :: nb,nc
  real(kind=8), dimension(n,n) :: f,p,a,pden,nbinv
  integer, dimension(n+2,n+2) :: s2
  real(kind=8), dimension(n+2,n+2) :: f2,f2s2

  !Initial condition and r allocation (does not need to be parallelized)
  s=1
  j1 = (n+1)/2
  s(j1,j1,:) = 0
  allocate(r(n,n))
  !-----------------------------------------------------------------------------
  !n2inv is the reciprocal of total number of villages in the grid (N*N)
  n2inv = 1.d0/dble(n*n)

  !s2 is matrix s with boundary of zeros around it, makes it easier to sum around
  !f2 is matrix f with boundary of zeros around it, makes it easier to sum around
  !$OMP parallel do collapse(2)
  do i4=1,n+2
    do j4=1,n+2
    s2(i4,j4)=0
    f2(i4,j4)=0.d0
    end do
  end do
  !$OMP end parallel do

  !Number of neighbours for inside points = 8
  !Number of neighbours for points on boundary = 5
  !Number of neighbours for corners = 3
  nb(1,1) = 3
  nb(1,n) = 3
  nb(n,1) = 3
  nb(n,n) = 3

  !$OMP parallel do collapse(2)
  do i5=2,n-1
    do j5=2,n-1
    nb(i5,j5) = 8
    nb(1,i5) = 5
    nb(n,i5) = 5
    nb(i5,1) = 5
    nb(i5,n) = 5
    end do
  end do
  !$OMP end parallel do

  !Reciprocal of the total number of neighbours
  !$OMP parallel do collapse(2)
  do i6=1,n
    do j6=1,n
    nbinv(i6,j6) = 1.d0/nb(i6,j6)
    end do
  end do
  !$OMP end parallel do

  !--------------Finished Problem Set Up------------------------

  !-----------------Time Marching-------------------------------
    do i3=1,nt
      call random_number(r) !Random numbers used to update s every time step

      !Set up coefficients for fitness score calculation in matrix a
      a = 1
      where(s(:,:,i3)==0)
        a=tr_b
      end where

      !Set inner square of s2 to be s
      s2(2:n+1,2:n+1) = s(:,:,i3)

      !Counts number of C neighbours for every village, done by overlapping section of matrix s2
      nc = s2(1:n,1:n) + s2(1:n,2:n+1) + s2(1:n,3:n+2) + &
           s2(2:n+1,1:n)                  + s2(2:n+1,3:n+2) + &
           s2(3:n+2,1:n)   + s2(3:n+2,2:n+1)   + s2(3:n+2,3:n+2)

      !Calculate fitness matrix f
      !C villages get 1pt for every C neighbour (a=1), 0pt for every M neighbour
      !M villages get bpt for every C neighbour (a=b), ept for every M neighbour
      f = nc*a
      where(s(:,:,i3)==0)
        f = f + (nb-nc)*tr_e
      end where
      f = f*nbinv

      !Set inner square of f2 to be f
      f2(2:n+1,2:n+1) = f

      !f contains fitness scores for C and M
      !f2s2 contains all of the fitness scores for just C
      f2s2 = f2*s2

      !Total fitness of collaborators in community
      p = f2s2(1:n,1:n) + f2s2(1:n,2:n+1) + f2s2(1:n,3:n+2) + &
           f2s2(2:n+1,1:n) + f2s2(2:n+1,2:n+1)  + f2s2(2:n+1,3:n+2) + &
          f2s2(3:n+2,1:n)   + f2s2(3:n+2,2:n+1)   + f2s2(3:n+2,3:n+2)

      !Total fitness of all members of coommunity
      pden = f2(1:n,1:n) + f2(1:n,2:n+1) + f2(1:n,3:n+2) + &
              f2(2:n+1,1:n) + f2(2:n+1,2:n+1)  + f2(2:n+1,3:n+2) + &
              f2(3:n+2,1:n)   + f2(3:n+2,2:n+1)   + f2(3:n+2,3:n+2)

      !Final probability matrix P_c
      !Final probability matrix P_m = 1-P_c
      p = (p/pden)*tr_g + 0.5d0*(1.d0-tr_g)

      !Set new affiliations based on probability matrix and random numbers stored in r
      !For each point if corresponding random probability is less than the P_c:
      !change to a C, if not then change to M
      s(:,:,i3+1) = 0
      where (r(:,:)<=p)
        s(:,:,i3+1) = 1
      end where
    end do

deallocate(r)

end subroutine simulate3_omp

!-----------------------
end module tribes
