"""Manlin Chawla 01205586"""

"""M3C 2018 Homework 3
Contains five functions:
    plot_S: plots S matrix -- use if you like
    simulate2: Simulate tribal competition over m trials. Return: all s matrices at final time
        and fc at nt+1 times averaged across the m trials.
    performance: To be completed -- analyze and assess performance of python, fortran, and fortran+openmp simulation codes
    analyze: To be completed -- analyze influence of model parameter, g
    visualize: To be completed -- generate animation illustrating "interesting" tribal dynamics
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import time
from scipy import stats
from scipy.optimize import curve_fit
from m1 import tribes as tr
#assumes that hw3_dev.f90 has been compiled with: f2py3 --f90flags='-fopenmp' -c hw3_dev.f90 -m m1 -lgomp
#May also use scipy and time modules as needed


def plot_S(S):
    """Simple function to create plot from input S matrix
    """
    ind_s0 = np.where(S==0) #C locations
    ind_s1 = np.where(S==1) #M locations
    plt.plot(ind_s0[1],ind_s0[0],'rs')
    plt.plot(ind_s1[1],ind_s1[0],'bs')
    plt.show()
    return None
#-------------------------------------------------------------------------------
def simulate2(N,Nt,b,e,g,m):
    """Simulate m trials of C vs. M competition on N x N grid over
    Nt generations. b, e, and g are model parameters
    to be used in fitness calculations.
    Output: S: Status of each gridpoint at end of simulation, 0=M, 1=C
            fc_ave: fraction of villages which are C at all Nt+1 times
                    averaged over the m trials
    """
    #Set initial condition
    S  = np.ones((N,N,m),dtype=int) #Status of each gridpoint: 0=M, 1=C
    j = int((N-1)/2)
    S[j,j,:] = 0
    N2inv = 1./(N*N)

    fc_ave = np.zeros(Nt+1) #Fraction of points which are C
    fc_ave[0] = S.sum()

    #Initialize matrices
    NB = np.zeros((N,N,m),dtype=int) #Number of neighbors for each point
    NC = np.zeros((N,N,m),dtype=int) #Number of neighbors who are Cs
    S2 = np.zeros((N+2,N+2,m),dtype=int) #S + border of zeros
    F = np.zeros((N,N,m)) #Fitness matrix
    F2 = np.zeros((N+2,N+2,m)) #Fitness matrix + border of zeros
    A = np.ones((N,N,m)) #Fitness parameters, each of N^2 elements is 1 or b
    P = np.zeros((N,N,m)) #Probability matrix
    Pden = np.zeros((N,N,m))
    #---------------------

    #Calculate number of neighbors for each point
    NB[:,:,:] = 8
    NB[0,1:-1,:],NB[-1,1:-1,:],NB[1:-1,0,:],NB[1:-1,-1,:] = 5,5,5,5
    NB[0,0,:],NB[-1,-1,:],NB[0,-1,:],NB[-1,0,:] = 3,3,3,3
    NBinv = 1.0/NB
    #-------------

    #----Time marching-----
    for t in range(Nt):
        R = np.random.rand(N,N,m) #Random numbers used to update S every time step

        #Set up coefficients for fitness calculation
        A = np.ones((N,N,m))
        ind0 = np.where(S==0)
        A[ind0] = b

        #Add boundary of zeros to S
        S2[1:-1,1:-1,:] = S

        #Count number of C neighbors for each point
        NC = S2[:-2,:-2,:]+S2[:-2,1:-1,:]+S2[:-2,2:,:]+S2[1:-1,:-2,:] + S2[1:-1,2:,:] + S2[2:,:-2,:] + S2[2:,1:-1,:] + S2[2:,2:,:]

        #Calculate fitness matrix, F----
        F = NC*A
        F[ind0] = F[ind0] + (NB[ind0]-NC[ind0])*e
        F = F*NBinv
        #-----------

        #Calculate probability matrix, P-----
        F2[1:-1,1:-1,:]=F
        F2S2 = F2*S2
        #Total fitness of cooperators in community
        P = F2S2[:-2,:-2,:]+F2S2[:-2,1:-1,:]+F2S2[:-2,2:,:]+F2S2[1:-1,:-2,:] + F2S2[1:-1,1:-1,:] + F2S2[1:-1,2:,:] + F2S2[2:,:-2,:] + F2S2[2:,1:-1,:] + F2S2[2:,2:,:]

        #Total fitness of all members of community
        Pden = F2[:-2,:-2,:]+F2[:-2,1:-1,:]+F2[:-2,2:,:]+F2[1:-1,:-2,:] + F2[1:-1,1:-1,:] + F2[1:-1,2:,:] + F2[2:,:-2,:] + F2[2:,1:-1,:] + F2[2:,2:,:]

        P = (P/Pden)*g + 0.5*(1.0-g) #probability matrix
        #---------

        #Set new affiliations based on probability matrix and random numbers stored in R
        S[:,:,:] = 0
        S[R<=P] = 1

        fc_ave[t+1] = S.sum()
        #----Finish time marching-----

    fc_ave = fc_ave*N2inv/m

    return S,fc_ave
#-------------------------------------------------------------------------------
def performance(figurenum,display=False):
    """Assess performance of simulate2, simulate2_f90, and simulate2_omp
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    """
    """Performance(1): This figure is a plot comparing the runtime of the Python,
    Fortran and Fortran+OpenMP implementation as N is increased. To generate this
    plot I have varied N from 7 to 75 and plotted the runtime at each odd N in this
    range. I have set the variables nt=100, b=1.1, e=0.01, g=0.95, m=100 and set
    the number of threads to 2. To visualize the trends between the runtime and N,
    I have fitted a regression line to each set of points.

    The plots show that as N increases, as expected each implementation takes longer
    to run. Overall the Python implementation (simulate2) is the slowest out of the
    three implementations and the Fortran implementation (simulate2_f90) is slower
    than the parallelized version (simulate2_omp).

    For small values of N such as N=7 all three implementations have a very similar
    run time. However, as N is increased the runtime for the Python implementation
    rapidly diverges from the runtimes of the Fortran and Fortran+OpenMP implementations.
    The runtimes for the Fortran implementation also increase but at a much shallower
    slope than the Python implementation. The Fortran implementation takes almost 1
    second to run when N=75, whereas the Python implementation takes almost 8 seconds
    to run for the same N. Likewise, the Fortran+OpenMP implementation runtimes also
    increase but at an even shallower slope the Fortran implementation and there are
    only marginal increases in runtimes as N increases. The runtime reaches almost 0.5
    seconds when N=75. This is due to parallelization of the code and the computations
    are split over 2 threads making the code overall faster.

    I found that a relatively good line of fit to the data was a quadratic. This makes
    sense as the parameter N defines the N x N grid of villages - as N is increased the
    calculations involved also increase by a factor of N squared. There is some fluctuation
    but this can be eliminated by running the functions again over the same range of N
    values multiple times and then taking an average. This would of course require much
    more time to produce the plot. To improve the plot I would run the plot for larger
    values of M to confirm the trend and get a better line of fit and to check if the
    quadratic line fits the data as N increases.
    --------------------------------------------------------------------------------------
    Performance(2): This figure is a plot comparing the runtime of the Python, Fortran
    and Fortran+OpenMP implementation as Nt is increased. To generate this plot I have
    varied Nt from 1 to 250 and plotted the runtime at each Nt in this range. I have set
    the variables n=21, b=1.1, e=0.01, g=0.95, m=100 and set the number of threads to 2.
    To visualize the trends between the runtime and Nt, I have fitted a regression line
    to each set of points.

    The plots show that as Nt increases, as expected each implementation takes longer to
    run. Overall the Python implementation (simulate2) is the slowest out of the three
    implementations and the Fortran implementation (simulate2_f90) is slower than the
    parallelized version (simulate2_omp) similar to the last plot. I found that a relatively
    good line of fit to the data was a linear regression line.

    Similar to the last plot all three implementations have a similar runtime for small
    values of Nt but the Python implementation rapidly increases whereas the Fortran and
    Fortran+OpenMP implementations runtimes increase at a shallower gradient. The
    Fortran+OpenMP implementation regression line has almost half the gradient as the
    Fortran regression line. This is since the parallelized code distributes the m trials
    across 2 threads halving the time of the computation.
    ---------------------------------------------------------------------------------------
    Performance(3): This figure is a plot comparing the runtime of the Python, Fortran
    and Fortran+OpenMP implementation as m (the number of time steps) is increased.
    To generate this plot I have varied the m from 10 to 400 and plotted the runtime for
    m increasing in increments of 10 in this range.I have set the variables n=21, nt=100,
    b=1.1, e=0.01, g=0.95 and set the number of threads to 2. To visualize the trends
    between the runtime and m, I have fitted a linear regression line to each set of points.
    This plot is very similar to the last plot and shows the same type of trend. Again,
    the Fortran+OpenMP implementation regression line has almost half the gradient as the
    Fortran regression line. This is since the parallelized code distributes the m trials
    across 2 threads halving the time of the computation.
    -----------------------------------------------------------------------------------------

    Performance(4): This figure is a plot comparing the Python, Fortran implementations to
    the Fortran+OpenMP implementation as a ratio. In particular this plot shows the ratio
    for the runtimes as m increases, with the variables set to the same as in the performance(3)
    plot. The solid lines across each set of points represents the mean ratio for that implementation.
    The previous three plots show that the Fortran+OMP implementation takes the least amount
    of time to run. On average the Fortran implementation takes twice as long to run as shown
    by the mean ratio being at 2.0. This is because simulate2_omp run is parallelized over 2 threads
    whereas simulate2_f90 is serial code so takes twice as long. On average the Python implementation
    is 10 times longer than the parallelized code, this is due to the combined fact that this is
    serial code and in an interpreted language which is considerably slower than a compiled
    language. There is some fluctuation, if I had more time to generate the plots I would eliminate
    this by running the functions again over the same range of m values multiple times and
    then taking an average of the runtime before calculating the ratio."""


    #Plot:Time taken for each function to run as N increases
    #N (x-axis), Runtime (y-axis)
    if figurenum==1:
        #Set parameters
        nt=100
        b=1.1
        e=0.01
        g=0.95
        m=100

        #Set parameters to use in Fortran implementations
        tr.tr_b=b
        tr.tr_e=e
        tr.tr_g=g
        tr.numthreads=2

        #Vectorize values of n
        nvalues=np.linspace(7,75,35).astype(int)

        #Preallocate vectors for runtime of each implementation
        runtime_py=np.zeros(35)
        runtime_for=np.zeros(35)
        runtime_omp=np.zeros(35)

        #Iterate over n, time each implementation for each value of m
        for i, n in enumerate(nvalues):
            print(n)
            #Runtime for Python implementation
            t0=time.time()
            simulate2(n,nt,b,e,g,m)
            t1=time.time()
            runtime_py[i]=t1-t0

            #Runtime for Fortran implementation
            t2=time.time()
            tr.simulate2_f90(n,nt,m)
            t3=time.time()
            runtime_for[i]=t3-t2

            #Runtime for Fortran+OpenMP implementation
            t4=time.time()
            tr.simulate2_omp(n,nt,m)
            t5=time.time()
            runtime_omp[i]=t5-t4

        #Define quadratic function for curve fitting using scipy.optimize.curve_fit
        def func(x,a,b,c):
            return a*(x**2)+b*x+c

        #Regression line for Python implementation
        para1=curve_fit(func,nvalues,runtime_py)
        [a1,b1,c1]=para1[0]
        line1=a1*(nvalues**2)+b1*(nvalues)+c1
        #Regression line for Fortran implementation
        para2=curve_fit(func,nvalues,runtime_for)
        [a2,b2,c2]=para2[0]
        line2=a2*(nvalues**2)+b2*(nvalues)+c2
        #Regression line for Fortran+OpenMP implementation
        para3=curve_fit(func,nvalues,runtime_omp)
        [a3,b3,c3]=para3[0]
        line3=a3*(nvalues**2)+b3*(nvalues)+c3

        #Plot of figure
        plt.hold(True)
        plt.plot(nvalues,runtime_py,'.',nvalues,line1,c='r',label='Python implementation')
        plt.plot(nvalues,runtime_for,'.',nvalues,line2,c='b',label='Fortran implementation')
        plt.plot(nvalues,runtime_omp,'.',nvalues,line3,c='c',label='Fortran+OMP implementation')
        plt.xlabel('N')
        plt.ylabel('Runtime (s)')
        plt.title('Manlin Chawla: performance(1) \n Runtime of each implementation as N increase')
        plt.legend()
        plt.hold(False)
        #plt.show()

        if display==True:
            plt.show()

        #Prints explicit form of the regression lines in the terminal
        print("Python Implementation line fit: runtime="+str(a1)+"*n^2"+"+"+str(b1)+"*n+"+str(c1))
        print("Fortran Implementation line fit: runtime="+str(a2)+"*n^2"+"+"+str(b2)+"*n+"+str(c2))
        print("Fortran=OpenMP line fit: runtime="+str(a3)+"*n^2"+"+"+str(b3)+"*n+"+str(c3))

    #Plot:Time taken for each function to run as Nt increases
    #Nt (x-axis), Runtime (y-axis)
    if figurenum==2:
        #Set parameters
        n=21
        b=1.1
        e=0.01
        g=0.95
        m=100

        #Set parameters to use in Fortran implementations
        tr.tr_b=b
        tr.tr_e=e
        tr.tr_g=g
        tr.numthreads=2

        #Vectorize values of nt
        ntvalues=np.linspace(1,250,250).astype(int)

        #Preallocate vectors for runtime of each implementation
        runtime_py=np.zeros(250)
        runtime_for=np.zeros(250)
        runtime_omp=np.zeros(250)

        #Iterate over n, time each implementation for each value of m
        for i, nt in enumerate(ntvalues):
            print(nt)
            #Runtime for Python implementation
            t0=time.time()
            simulate2(n,nt,b,e,g,m)
            t1=time.time()
            runtime_py[i]=t1-t0

            #Runtime for Fortran implementation
            t2=time.time()
            tr.simulate2_f90(n,nt,m)
            t3=time.time()
            runtime_for[i]=t3-t2

            #Runtime for Fortran+OpenMP implementation
            t4=time.time()
            tr.simulate2_omp(n,nt,m)
            t5=time.time()
            runtime_omp[i]=t5-t4

        #Regression line for Python implementation
        slope1, intercept1, r_value, p_value, std_err=stats.linregress(ntvalues,runtime_py)
        line1=slope1*ntvalues+intercept1
        #Regression line for Fortran implementation
        slope2, intercept2, r_value, p_value, std_err =stats.linregress(ntvalues,runtime_for)
        line2=slope2*ntvalues+intercept2
        #Regression line for Fortran+OpenMP implementation
        slope3, intercept3, r_value, p_value, std_err =stats.linregress(ntvalues,runtime_omp)
        line3=slope3*ntvalues+intercept3

        #Plot of figure
        plt.hold(True)
        plt.plot(ntvalues,runtime_py,'.',ntvalues,line1,c='r',label='Python implementation')
        plt.plot(ntvalues,runtime_for,'.',ntvalues,line2,c='b',label='Fortran implementation')
        plt.plot(ntvalues,runtime_omp,'.',ntvalues,line3,c='c',label='Fortran+OMP implementation')
        plt.xlabel('Nt')
        plt.ylabel('Runtime (s)')
        plt.title('Manlin Chawla: performance(2) \n Runtime of each implementation as the number of years (Nt) increase')
        plt.legend()
        plt.hold(False)
        #plt.show()

        if display==True:
            plt.show()

        #Prints explicit form of the regression lines in the terminal
        print("Python Implementation line fit: runtime="+str(slope1)+"*nt"+"+"+str(intercept1))
        print("Fortran Implementation line fit: runtime="+str(slope2)+"*nt"+"+"+str(intercept2))
        print("Fortran+OpenMP Implementation line fit: runtime="+str(slope3)+"*nt"+"+"+str(intercept3))

    #Plot:Time taken for each function to run as M increases
    #M (x-axis), Runtime (y-axis)
    if figurenum==3:
        #Set parameters
        n=21
        nt=100
        b=1.1
        e=0.01
        g=0.95

        #Set parameters to use in Fortran implementations
        tr.tr_b=b
        tr.tr_e=e
        tr.tr_g=g
        tr.numthreads=2

        #Vectorize values of m
        mvalues=np.linspace(10,400,79).astype(int)

        #Preallocate vectors for runtime of each implementation
        runtime_py=np.zeros(79)
        runtime_for=np.zeros(79)
        runtime_omp=np.zeros(79)

        #Iterate over m, time each implementation for each value of m
        for i, m in enumerate(mvalues):
            print(m)
            #Runtime for Python implementation
            t0=time.time()
            simulate2(n,nt,b,e,g,m)
            t1=time.time()
            runtime_py[i]=t1-t0

            #Runtime for Fortran implementation
            t2=time.time()
            tr.simulate2_f90(n,nt,m)
            t3=time.time()
            runtime_for[i]=t3-t2

            #Runtime for Fortran+OpenMP implementation
            t4=time.time()
            tr.simulate2_omp(n,nt,m)
            t5=time.time()
            runtime_omp[i]=t5-t4

        #Regression line for Python implementation
        slope1, intercept1, r_value, p_value, std_err=stats.linregress(mvalues,runtime_py)
        line1=slope1*mvalues+intercept1
        #Regression line for Fortran implementation
        slope2, intercept2, r_value, p_value, std_err =stats.linregress(mvalues,runtime_for)
        line2=slope2*mvalues+intercept2
        #Regression line for Fortran+OpenMP implementation
        slope3, intercept3, r_value, p_value, std_err =stats.linregress(mvalues,runtime_omp)
        line3=slope3*mvalues+intercept3

        #Plot of figure
        plt.hold(True)
        plt.plot(mvalues,runtime_py,'.',mvalues,line1,c='r',label='Python implementation')
        plt.plot(mvalues,runtime_for,'.',mvalues,line2,c='b',label='Fortran implementation')
        plt.plot(mvalues,runtime_omp,'.',mvalues,line3,c='c',label='Fortran+OMP implementation')
        plt.xlabel('Number of trials (m)')
        plt.ylabel('Runtime (s)')
        plt.title('Manlin Chawla: performance(3) \n Runtime of each implementation as the number of trials (m) increases ')
        plt.legend()
        plt.hold(False)
        #plt.show()

        if display==True:
            plt.show()

        #Prints explicit form of the regression lines in the terminal
        print("Python Implementation line fit: runtime="+str(slope1)+"*m"+"+"+str(intercept1))
        print("Fortran Implementation line fit: runtime="+str(slope2)+"*m"+"+"+str(intercept2))
        print("Fortran+OpenMP Implementation line fit: runtime="+str(slope3)+"*m"+"+"+str(intercept3))

    #Plot:Ratio of the runtimes
    #M (x-axis), Ratio of runtime (y-axis)
    if figurenum==4:
        #Set parameters
        n=21
        nt=100
        b=1.1
        e=0.01
        g=0.95

        #Set parameters to use in Fortran implementations
        tr.tr_b=b
        tr.tr_e=e
        tr.tr_g=g
        tr.numthreads=2

        #Vectorize values of m
        mvalues=np.linspace(10,500,50).astype(int)

        #Preallocate vectors for runtime of each implementation
        runtime_py=np.zeros(50)
        runtime_for=np.zeros(50)
        runtime_omp=np.zeros(50)
        runtimeratio_pyave=np.zeros(50)
        runtimeratio_forave=np.zeros(50)

        #Iterate over m, time each implementation for each value of m
        for i, m in enumerate(mvalues):
            print(m)
            #Runtime for Python implementation
            t0=time.time()
            simulate2(n,nt,b,e,g,m)
            t1=time.time()
            runtime_py[i]=t1-t0

            #Runtime for Fortran implementation
            t2=time.time()
            tr.simulate2_f90(n,nt,m)
            t3=time.time()
            runtime_for[i]=t3-t2

            #Runtime for Fortran+OpenMP implementation
            t4=time.time()
            tr.simulate2_omp(n,nt,m)
            t5=time.time()
            runtime_omp[i]=t5-t4

        #Divide to get the rntime (satio of runtimes
        runtimeratio_py=runtime_py/runtime_omp
        runtimeratio_pyave[:]=runtimeratio_py.mean()
        runtimeratio_for=runtime_for/runtime_omp
        runtimeratio_forave[:]=runtimeratio_for.mean()

        #Plot of figure
        plt.hold(True)
        plt.plot(mvalues,runtimeratio_py,'.',mvalues,runtimeratio_pyave,c='r',label='Python implementation')
        plt.plot(mvalues,runtimeratio_for,'.',mvalues,runtimeratio_forave,c='b',label='Fortran implementation')
        plt.xlabel('Number of trials (m)')
        plt.ylabel('Ratio')
        plt.title('Manlin Chawla: performance(4) \n Ratio of runtime of each implementation compared to the Fortran+OMP implementation, as M increases')
        plt.legend()
        plt.hold(False)
        #plt.show()

        if display==True:
            plt.show()

    return None #Modify as needed
#-------------------------------------------------------------------------------
def analyze(figurenum,display=False):
    """Analyze influence of model parameter, g.
    Modify the contents of the tuple, input, as needed
    When display is True, figures equivalent to those
    you are submitting should be displayed
    """

    """Analyze(1): This figure is a plot showing the fc_ave after each time step
    Nt for different values of gamma. I have set the variables n=51, nt=100, m=100,
    b=1.1, e=0.01 and set the number of threads to 2.

    As gamma increases, more time steps are needed before the fc_ave's settle to
    a value. For smaller values of gamma there is a steep decrease but the fc_ave
    converge quite quickly to a value. As gamma increases there is a shallower
    decrease but the fc_ave's take longer to converge and settle at a lower value.
    This could mean that as gamma increases the fc_ave's are more likely converge
    towards zero meaning htat the entire grid becomes Mercernary villages. However
    gamma=1 doesn't follow this trend and has a relatively higher fc_ave (closer
    outwards 1) and drops slightly as Nt approaches 100.

    Analyze(2):This figure is a plot showing the final fc_ave as gamma is varied.
    I have set the variables n=51, nt=100, m=100, e=0.01 and set the number of
    threads to 2. Each line seems to reach its minimum for a high value of gamma before
    all of the final fc_ave's tend to 1. This is similar to the previous plot where the
    final fc_ave for gamma=1.0 remains close to 1. Also the higher the value of b the
    higher the values of the final fc_ave are throughout.

    Analyze(3): This figure is a plot showing the final fc_ave as beta is varied.
    I have set the variables n=51, nt=100, m=100, e=0.01 and set the number of
    threads to 2. As b increases the final fc_ave's decrease for all of the lines and
    eventually converge to a value. As gamma increases the value of the final fc_ave that
    the system settles to is lower, meaning there is trend towards S converging to
    a state of all Mercernaries. Another interesting feature is that as gamma increases
    each line settles to a value of fc_ave for a smaller value of b. An interesting
    case is gamma=1.0 which has fc_ave at a higher value than all other values of
     gamma exhibit. However it also displays chaotic behaviour a b increases past 1.1."""

    #Plot:average of fc over 100 simulations for different values of gamma
    #X-axis=Nt, Y-axis=f_c
    if figurenum==1:
        #Set parameters
        n=51
        nt=100
        m=100

        #Set parameters to use in Fortran implementations
        tr.tr_b=1.1
        tr.tr_e=0.01
        tr.numthreads=2

        #Vectorize values of g
        gvalues=np.linspace(0.8,1,11)

        #Iterate over g, fc_ave for each value of g
        plt.hold(True)
        for g in gvalues:
            print(g)
            tr.tr_g=g
            S,fc_ave=tr.simulate2_omp(n,nt,m)

            #Plot for each value of gamma
            plt.plot(fc_ave,label='gamma='+str(g))

        #Plotting format
        plt.xlabel('Year (Nt)')
        plt.ylabel('fc_ave')
        plt.title('Manlin Chawla: analyze(1) \n Average of fc over 100 simulations for different values of gamma')
        plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot:average of fc over 100 simulations for different values of gamma and beta
    #X-axis=Nt, Y-axis=f_c
    if figurenum==2:

        #Set parameters to use in Fortran implementation
        n=51
        nt=100
        m=100
        tr.tr_e=0.01
        tr.numthreads=2

        #Vectorize values of g
        gvalues=np.linspace(0.8,1,11)
        bvalues = np.linspace(1,1.5,11); bvalues[0]=1.01

        #Preallocate vector for last value of fc_ave
        fc_avefinal=np.zeros(len(gvalues))

        #Iterate over b, plot a line for every b in bvalues
        plt.hold(True)
        for b in bvalues:
            #Set b to use in Fortran implementations
            tr.tr_b=b

            #Iterate over g, get last value of fc when g is varied
            for i, g in enumerate(gvalues):
                #Set g to use in Fortran implementations
                tr.tr_g=g
                S,fc_ave=tr.simulate2_omp(n,nt,m)
                fc_avefinal[i]=fc_ave[nt]

            #Plot for each value of b
            plt.plot(gvalues,fc_avefinal,label='b='+str(b))

        #Plotting format
        plt.xlabel('Gamma (g)')
        plt.ylabel('fc_ave')
        plt.title('Manlin Chawla: analyze(2) \n Average of fc over 100 simulations for different values of gamma')
        plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    #Plot:average of fc over 100 simulations for different values of gamma and beta
    #X-axis=Nt, Y-axis=f_c
    if figurenum==3:

        #Set parameters to use in Fortran implementation
        n=51
        nt=100
        m=100
        tr.tr_e=0.01
        tr.numthreads=2

        #Vectorize values of g
        gvalues=np.linspace(0.8,1,11)
        bvalues = np.linspace(1,1.5,101); bvalues[0]=1.01

        #Preallocate vector for last value of fc_ave
        fc_avefinal=np.zeros(len(bvalues))

        #Iterate over g, plot a line for every g in gvalues
        plt.hold(True)
        for g in gvalues:
            print(g)
            #Set g to use in Fortran implementations
            tr.tr_g=g

            #Iterate over b, get last value of fc when b is varied
            for i, b in enumerate(bvalues):
                #Set g to use in Fortran implementations
                tr.tr_b=b
                S,fc_ave=tr.simulate2_omp(n,nt,m)
                fc_avefinal[i]=fc_ave[nt]

            #Plot for each value of g
            plt.plot(bvalues,fc_avefinal,label='gamma='+str(g))

        #Plotting format
        plt.xlabel('b')
        plt.ylabel('fc_ave')
        plt.title('Manlin Chawla: analyze(3) \n Average of fc over 100 simulations for different values of gamma')
        plt.legend()
        plt.hold(False)

        if display==True:
            plt.show()

    return None #Modify as needed
#-------------------------------------------------------------------------------
def visualize():
    """Generate an animation illustrating the evolution of
        villages during C vs M competition
    """
    #Set parameters for Fortran+OpenMP implementation
    n=21
    nt=100
    m=100
    tr.tr_b=1.45
    tr.tr_e=0.01
    tr.tr_g=0.85
    tr.numthreads=2

    #Open clear figure
    fig = plt.figure()
    #Call subroutine simulate3 (see comments in docstring) from the hw3.f90 file
    #Output of simulate3 is state matrix S of dimension n x n x nt+1
    #Output shows S after each year (nt)
    S=tr.simulate3_omp(n,nt)

    #Updates each fram of the animation
    def updatefig(i):
        print("iteration=",i)
        nt=i
        Splot=S[:,:,i]

        #Plot of S after each yea, code from function plot_S
        ind_s0 = np.where(Splot==0) #C locations
        ind_s1 = np.where(Splot==1) #M locations
        image=plt.plot(ind_s0[1],ind_s0[0],'rs',ind_s1[1],ind_s1[0],'bs')
        plt.title('Manlin Chawla: visualize() \n N=21 Nt=100 m=100 b=1.45 e=0.01 g=0.85')
        plt.axis('off')
        return image,

    #Create animation
    ani = animation.FuncAnimation(fig, updatefig, frames=100,interval=600,repeat=False)
    #Save animation
    ani.save('hw3movie.mp4',writer="ffmpeg")

    return None #Modify as needed

if __name__ == '__main__':
      #Set display=False in funcion performance before saving final figures
      output_p = performance(1)
      plt.savefig('hw31.png', bbox_inches="tight")
      plt.clf()
      print('performance(1) plot saved')

      output_p = performance(2)
      plt.savefig('hw32.png', bbox_inches="tight")
      plt.clf()
      print('performance(2) plot saved')

      output_p = performance(3)
      plt.savefig('hw33.png', bbox_inches="tight")
      plt.clf()
      print('performance(3) plot saved')

      output_p = performance(4)
      plt.savefig('hw34.png', bbox_inches="tight")
      plt.clf()
      print('performance(4) plot saved')

      output_a = analyze(1)
      plt.savefig('hw35.png', bbox_inches="tight")
      plt.clf()
      print('analyze(1) plot saved')

      output_a = analyze(2)
      plt.savefig('hw36.png', bbox_inches="tight")
      plt.clf()
      print('analyze(2) plot saved')

      output_a = analyze(3)
      plt.savefig('hw37.png', bbox_inches="tight")
      plt.clf()
      print('analyze(3) plot saved')
