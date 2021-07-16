!Manlin Chawla 01205586

! This is a main program which can be used with the tribes module
! if and as you wish as you develop your codes.
! It reads the problem parameters from a text file, data.in, which must be created.
!
! The subroutine simulate2_f90 is called, and output is written to the text files,
! s.txt and fc_ave.txt which can be read in Python using np.loadtxt (see below)
!
! You should not submit this code with your assignment.
! To compile: gfortran -fopenmp -O3 -o main.exe hw3_dev.f90 hw3_main.f90
program hw3_main
  use tribes
  implicit none
  integer :: n,nt,m,i1,j1
  real(kind=8), allocatable, dimension(:) :: fc_ave
  integer, allocatable, dimension(:,:,:) :: s

  !Read in problem parameters from text file, data.in
  open(unit=11,file='data.in')
  read(11,*) n !n x n villages
  read(11,*) nt !number of time steps
  read(11,*) tr_b !model parameters
  read(11,*) tr_e
  read(11,*) tr_g
  read(11,*) m !number of trials
  read(11,*) numthreads !not used below
  close(11)

  allocate(fc_ave(nt+1),s(n,n,m))

  call simulate2_omp(n,nt,m,s,fc_ave)
  !call simulate3_omp(n,nt,s)

  !load in python using fc_ave = np.loadtxt('fc_ave.txt')
  open(unit=12,file='fc_ave.txt')
  do i1=1,nt+1
    write(12,*) fc_ave(i1)
  end do
  close(12)

  !load in python using: s = np.loadtxt('s.txt')
  !                      s = s.reshape(n,n,m)
  !with n and m set appropriately
  open(unit=13,file='s.txt')
  do j1=1,m
    do i1=1,n
      write(13,'(1000I4)') s(i1,:,j1)
    end do
  end do
  close(13)


end program hw3_main
