from __future__ import (absolute_import, division, #
                        print_function, unicode_literals)
exec(open("ground.py").read())
"""
sde.py
--------------
Noisy landmark image registration

Classes:
SDE, SDELin
MAP1, MAP2 (first-splitting prior with shooting and multi-shooting)
MAP3, MAP4 (second-splitting prior with...)
MAP5 (second-splitting prior, with mulitple landmark sets - untested)

TS, Jan 2016 (made compatible with vectorized hamiltonian.py)
Feb 2016 (factored as SDE for basics, and SDELin for linearisation)
"""
from timeit import default_timer as timer
import scipy.optimize as spo
import scipy.linalg as spla
from numpy import linalg as LA
# plotting
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
# mine
import utility
import hamiltonian
import diffeo
from diffeo import Diffeo
########################
########################
"""
Class:  SDEDiffeo(G)
Langevin eqn based on Green's fn G (see hamiltonian.py)

Provides: SDE, SDELin, MAPx
"""
class SDE(Diffeo):
    def __init__(self, G, report_flag=False):
        """
        SDE: initialise by providing Green's fn
        """
        if isinstance(G, hamiltonian.GaussGreen):
            Diffeo.__init__(self, G, report_flag)
        elif isinstance(G, Diffeo):
            Diffeo.__init__(self, G.G, report_flag)
            self.copy(G)
        else:
            assert False, print("numpty",type(G))
        # some defaults
        self._set_lam_sig(1.0, 0.0001,False)
        self.c=1 # data term coeff
        self.TOL=1e-3 # for gradient checker
    #
    def set_data_var(self, data_var, report_flag=True):
        """
        SDE: set data parameter
        """
        self.c=0.5/data_var
        if report_flag:
            print("Set data var = ",data_var)
            sys.stdout.flush()
    #
    def set_lam_beta(self, lam, beta, report_flag=True):
        """
        SDE: Set heat bath params (lam=dissipation, beta=inv temp)
        """
        sig=sqrt(2*lam/beta)
        self._set_lam_sig(lam,sig,report_flag)
    #
    def _set_lam_sig(self, lam, sig, report_flag=False):
        """
        SDE: Set heat bath params (lam=dissipation, sig=diffusion )
        Better to use set_lam_beta()
        """
        self.lam =lam
        self.sig =sig
        self.beta     =2*lam/sig**2
        self.half_beta=  lam/sig**2
        if report_flag:
            print("Set lambda = ", self.lam)
            print("    sigma  = ", self.sig )
            print("    beta   = ", self.beta) # inverse temperature
            sys.stdout.flush()

    #
    def fd_Hessian(self):
        """
        SDE: compute a Hessian approx at u0 via finite differences
        """
        print("Computing finite-difference approx to Hessian...")
        sys.stdout.flush()
        # extract data from uin
        delta=1e-6
        d=self.u0.size
        H=np.empty((d,d))
        start=timer()
        for i in range(d):
            e=np.zeros(d)
            e[i]=delta
            H[:,i]=(self.gradient(self.u0+e)
                    -self.gradient(self.u0-e))/(2*delta)
            #
        print("Run time %3.1f secs (Hessian approx)" % (timer()-start))
        sys.stdout.flush()
        return H
    #
    def _gradient_checker(self,dim):
        """
        SDE:
        """
        j=np.random.random_integers(dim-1)
        #
        delta=1e-8
        u0=np.random.normal(0,1,dim)
        F=self.objective(u0)
        e=np.zeros(u0.shape)
        e[j]=delta;
        J_fd=(self.objective(u0+e)-self.objective(u0-e))/(2*delta)
        J_an=self.gradient(u0)
        err=np.linalg.norm(J_fd-J_an[j])
        if(err>self.TOL*np.linalg.norm(J_an[j])):
            print("Gradient", J_an[j], "\n finite-difference approx", J_fd)
            print("checker error: ",err)
            assert False
        print("     Passed Gradient checker")
    #
    def _grad_data(self,P,Q,Q_data,no_steps):
        """
        SDE: return the gradient of the data term
        0.5 * c * l2-norm( Q(time 1) - Q_data)^2
        """
        DPP,DQP,DPQ,DQQ,P1,Q1=self._Jac_forward_all(P,Q,no_steps)
        tmp=Q1-Q_data
        grad_data_p = self._contract_t(tmp,DQP)
        grad_data_q = self._contract_t(tmp,DQQ)
        return self.c*np.vstack((grad_data_p,
                                 grad_data_q)).reshape((2,self.d,self.N))
    #
    def _Hess_data_GN(self,Ph,Qh,no_steps):
        """
        SDE: return the GN approx to Hessian of data-factor term
        0.5*c*|q(at time-T starting at (Ph,Qh))-Q_data|^2
        where T is determed by no_steps * timestep.

        does not need q_data to be computed/provided
        """
        DPP0,DQP0,DPQ0,DQQ0,P0,Q0=self._Jac_forward_all(Ph,Qh,no_steps)
        c_sqrt=sqrt(self.c)
        grad_data0_factor=np.zeros((self.d,self.N,2,self.d,self.N))
        grad_data0_factor[:,:,0,:,:]=c_sqrt*DQP0
        grad_data0_factor[:,:,1,:,:]=c_sqrt*DQQ0
        return   np.tensordot(grad_data0_factor,grad_data0_factor,
                               axes=([0,1],[0,1]))
        #

    #
    def _one_step_em(self, P, Q, dw):
        """
        SDE: Evaluate one step of Euler-Maruyama.
        dw provides Brownian increment
        return updated P,Q
        """
        HP=self.Dp(P, Q)
        HQ=self.Dq(P, Q)
        #
        Pn=P-(self.lam*HP+HQ)*self.dt+self.sig*dw
        Qn=Q+HP*self.dt
        return Pn, Qn
    #
    def sample_push_forward(self,Qr):
        """
        SDE: Sample the push-forward of Qr
        and store the path in Qpath,Ppath with indices
        [i,j,k] i=time, j=spatial dimension, k=particle number.

        return as a Diffeo object
        """
        #
        Q=np.copy(Qr)
        P=self.sample_Gibbs_for_p(Q,self.beta)
        #
        D=diffeo.Diffeo(self.G)
        D.N=Qr.shape[1]
        D.set_no_steps(self.no_steps)
        D._init_path()
        D.Qpath[0,:,:]=Q
        D.Ppath[0,:,:]=P

        dw=np.random.normal(0,self.dt, (self.d, Q.shape[1], self.no_steps) )
        for i in range(self.no_steps):
            P, Q=self._one_step_em(P, Q, dw[:,:,i])
            D.Qpath[i+1,:,:]=Q
            D.Ppath[i+1,:,:]=P
        return D
    #
    def add_sde_noise(self,Q0,no_sets):
        """
        Sample the push-forward map (no_sets times), with initial
        (P0,Q0) given, using the SDE defined by SDE
        """
        X=np.empty((no_sets, 2, Q0.shape[1]))
        for i in range(no_sets):
            D=self.sample_push_forward(Q0)
            X[i,:]=D.Qpath[-1,:,:]
        return utility.procrust1(X)
    #
    def _add_to(self,A,B,ind):
        """
        SDE: for 6-index tensors A and B,
        add entries from B to specified entries of A. Typically, B
        is smaller than A and this cannot be achieved directly
        """
        for i in range(len(ind)):
            A[ind,:,:,ind[i],:,:]+= B[:,:,:,i,:,:]
    #

#############################
class SDELin(SDE):
    #
    def set_prior_eps(self, var_):
        """
        SDELin: Set standard deviation for initial covariance
        for propogating, to create prior.
        """
        self.prior_epsilon2  = var_
        print("Set prior epsilon variance :",
              self.prior_epsilon2)
    #
    def set_lin_path(self, Ppath, Qpath):
        """
        SDELin: Set paths to linearise around.
        """
        assert(self.N==Qpath.shape[2])
        # linearisation path
        self.PLinPath = np.copy(Ppath)
        self.QLinPath = np.copy(Qpath)
        # initialise tensor for moment matrix
        s=np.array([2*self.d,self.N])
        self.Af_dt = np.zeros(s)
        self.Ab_dt = np.zeros(s)
        #
        self.B = np.zeros(np.concatenate((s,s)))
        self.matplus = np.zeros_like(self.B)
        self.matminus = np.zeros_like(self.B)
    #
    def _set_Af_dt(self,t):
        """
        SDELin:
        """
        # Sets the constant part of the linear SDE, looking forward
        PP=0.5*(self.PLinPath[t,:,:]+self.PLinPath[t+1,:,:])
        QQ=0.5*(self.QLinPath[t,:,:]+self.QLinPath[t+1,:,:])
        # q (bottom) part of A always zero
        const=-0.5*self.dt*self.lam # carry dt here
        self.Af_dt[:self.d,:] = const * self.Dp(PP,QQ)
    #
    def _set_Ab_dt(self,t):
        """
        SDELin:
        """
        # Sets the constant part of the linear SDE, looking backward
        PP=0.5*(self.PLinPath[t,:,:]+self.PLinPath[t-1,:,:])
        QQ=0.5*(self.QLinPath[t,:,:]+self.QLinPath[t-1,:,:])
        # q (bottom) part of A always zero
        const=-0.5*self.dt*self.lam # carry dt here
        self.Ab_dt[:self.d,:] = const * self.Dp(PP,QQ)
    #
    def _set_Bf(self,t):
        """
        SDELin:
        """
        # Sets the linear part of the affine SDE, looking forward
        PP=0.5*(self.PLinPath[t,:,:]+self.PLinPath[t+1,:,:])
        QQ=0.5*(self.QLinPath[t,:,:]+self.QLinPath[t+1,:,:])
        # Top left
        self.B[:self.d,:,:self.d,:] = -self.lam*self.Dpp(PP,QQ) -self.Dpq(PP,QQ)
        # Top right
        self.B[:self.d,:,self.d:,:] = -self.lam*self.Dpq(PP,QQ)-self.Dqq(PP,QQ)
        # Bottom left
        self.B[self.d:,:,:self.d,:] = self.Dpp(PP,QQ)
        # Bottom right
        self.B[self.d:,:,self.d:,:] = self.Dpq(PP,QQ)
    #
    def _set_Bb(self,t):
        """
        SDELin:
        """
        assert(t>0)
        # Sets the linear part of the affine SDE, looking backward
        PP=0.5*(self.PLinPath[t,:,:]+self.PLinPath[t-1,:,:])
        QQ=0.5*(self.QLinPath[t,:,:]+self.QLinPath[t-1,:,:])
        # Top left
        self.B[:self.d,:,:self.d,:]=-self.lam*self.Dpp(PP,QQ)+self.Dpq(PP,QQ)
        # Top right
        self.B[:self.d,:,self.d:,:]=-self.lam*self.Dpq(PP,QQ)+self.Dqq(PP,QQ)
        # Bottom left
        self.B[self.d:,:,:self.d,:]= -self.Dpp(PP,QQ)
        # Bottom right
        self.B[self.d:,:,self.d:,:]= -self.Dpq(PP,QQ)
    #
    def _set_Mplus(self,t):     #
        """
        SDELin: Define (I+h B), using B looking forward
        """
        self._set_Bf(t)
        self.matplus = self.B*self.dt
        # for i in range(2*self.d):
        #     for j in range(self.N):
        #         self.matplus[i,j,i,j] += 1
        # cute trick for above three lines :-)
        increment=(2*self.d*self.N)+1
        self.matplus.flat[::increment]+=1
    #
    def _set_Mminus(self,t):
        """
        SDELin: Define (I-h B), using B looking back
        """
        self._set_Bb(t)
        self.matminus = self.B*self.dt
        # for i in range(2*self.d):
        #     for j in range(self.N):
        #         self.matminus[i,j,i,j] += 1
        # cute trick again
        increment=(2*self.d*self.N)+1
        self.matminus.flat[::increment]+=1
    #
    def do_all(self,data_var):
        """
        SDELin: Find mean, covariance, and conditional distribution.
        """
        t0 = 0
        t1 = int( floor((self.no_steps-1)/2.))
        t2 = self.no_steps
        #
        print("SDELin: calculating mean and convariance...")
        # initialise
        start=timer()
#        self.InitialiseDistribution(t1,self.prior_epsilon2)
        self.initialise_distribution2(t1,self.prior_epsilon2,
                                     self.QLinPath[t1,:,:],
                                     self.PLinPath[t1,:,:])
        # compute mean and covariance
        self.set_path_dist_diagonal(t0,t1,t2)
        self.set_path_dist_non_diagonal(t0,t1,t2)
        # Compute covariance matrix from MomentMat
        delta=self.deltaPQMean.view()
        self.C = self.MomentMat - np.einsum('ace,bdf->adebcf',delta,delta)
        #
        self.condition_mat(data_var)
        end=timer()
        print("Run time %3.1f secs" % (end-start))
    #
    def initialise_distribution(self,t,variance):
        """
        SDELin: Set distribution of deltaPQMean and MomentMat at time t,
        with zero mean and input variance
        """
        s=np.array([self.no_steps+1,2*self.d,self.N])
        # set inital mean to zero
        self.deltaPQMean = np.zeros(s)
        # set inital covariance
        self.MomentMat = np.zeros(np.concatenate((s,s)))
        np.fill_diagonal(self.MomentMat[t,:,:,t,:,:],variance)
    #
    def initialise_distribution2(self,t,variance,Q,P):
        """
        SDELin: Set Gaussian distribution at t=1/2 by setting mean and
        second-moment matrix (not covariance, BEWARE) for delta =(p-plin, ...).

        For prior with mean delta = (-plin, 0) and cov =(Gmat, variance*I),
        need moment_mat=(Gmat+plin**2, variance)

        Here, Gmat is the covariance of Gibbs dist given Q
        """
        s=np.array([self.no_steps+1,2*self.d,self.N])
        # set inital mean to zero
        self.deltaPQMean = np.zeros(s)
        self.deltaPQMean[t,:self.d,:]=-P*0
        # set inital covariance
        self.MomentMat = np.zeros(np.concatenate((s,s)))
        # remember we are defining initial moment matrix
        for i in range(self.d,2*self.d): # position(Q) variables
            np.fill_diagonal(self.MomentMat[t,i,:,t,i,:],variance)
        # for i in range(self.d,s[0]):
        #     for j in range(s[1]):
        #         self.MomentMat[i,j,t,i,j,t]=self.prior_epsilon2+Q[i-self.d,j]**2
        # for mean p, mean 0 and gibbs distribtion
        CovP=self.Gibbs_cov_given_q(Q,self.beta)
        for i in range(self.d):
            self.MomentMat[t,i,:,t,i,:]=CovP#+outer(P[i,:],P[i,:])

    #
    def set_path_dist_diagonal(self,t0,t1,t2):
        """
        SDELin: Computes the mean and diagonal moment matrix.
        Diagonal means MomentMat[t,:,:,t,:,:]
        Runs BE from t1 to t0.
        Runs FE from t1 to t2, assuming t0<t1<t2.
        InitialiseDistribution should be called beforehand.
        """
        # seriously, let's abbreviate!
        p_range=range(self.d)
        particle_range=range(self.N)
        M=self.MomentMat
        delta=self.deltaPQMean
        Af_dt=self.Af_dt
        Ab_dt=self.Ab_dt
        noise_const=self.sig**2*self.dt
        #  Forward Euler
        for t in range(t1,t2):
            self._set_Af_dt(t)
            self._set_Mplus(t)
            Mp=self.matplus.view()
            # Mean
            delta[t+1,:,:]=np.add(self._contract(Mp, delta[t,:,:]),
                                  Af_dt)
            # Covariance diagonal
            M[t+1,:,:,t+1,:,:]=self._contract(Mp,self._contract_tt(
                M[t,:,:,t,:,:],Mp))
            M[t+1,:,:,t+1,:,:]+=self._outerproduct(Af_dt,delta[t+1,:,:])
            M[t+1,:,:,t+1,:,:]+=self._outerproduct(delta[t+1,:,:], Af_dt)
            M[t+1,:,:,t+1,:,:]-=self._outerproduct(Af_dt, Af_dt)
            # Add on BM increment
            for i in p_range:
                for j in particle_range:
                    M[t+1,i,j,t+1,i,j] += noise_const
        # Backward Euler
        for t in range(t1,t0,-1): # includes t1, excludes t0, runs backward
            self._set_Ab_dt(t)
            self._set_Mminus(t)
            Mm=self.matminus.view()
            # Mean
            delta[t-1,:,:] = np.add(self._contract(Mm, delta[t,:,:]), Ab_dt)
            # Covariance diagonal
            M[t-1,:,:,t-1,:,:] =self._contract(Mm,
                                                 self._contract_tt(
                                                     M[t,:,:,t,:,:], Mm))
            M[t-1,:,:,t-1,:,:]+=self._outerproduct(Ab_dt,
                                                   delta[t-1,:,:])
            M[t-1,:,:,t-1,:,:]+=self._outerproduct(delta[t-1,:,:],
                                                   Ab_dt)
            M[t-1,:,:,t-1,:,:]-=self._outerproduct(Ab_dt, Ab_dt)
            # Add on BM increment
            for i in p_range:
                for j in particle_range:
                    M[t-1,i,j,t-1,i,j] += noise_const
    #
    def set_path_dist_non_diagonal(self,t0,t1,t2):
        """
        SDELin: Computes the non-diagonal covariance
        after SetPathDistDiagonal
        """
        # Forward Euler
        for t in range(t1,t2+1):
            # lower-right off-diagonal block
            for j in range(t+1,t2+1):
                self._set_Af_dt(j-1)
                self._set_Mplus(j-1)
              #
                self._nondiag_fe_update(t,j-1)
        # Backward Euler
        for t in range(t1,t0-1,-1):
             # upper-left off-diagonal block
            for j in range(t-1,t0-1,-1):
                self._set_Ab_dt(j+1)
                self._set_Mminus(j+1)
               #
                self._nondiag_be_update(t,j+1)
        # remaining blocks, either by FE or BE; choose FE.
        for t in range(t1,t2):
             self._set_Af_dt(t)
             self._set_Mplus(t)
             for j in range(0,t1):
                 self._nondiag_fe_update(j,t)
    #
    def _nondiag_fe_update(self,j,t):
        """
        SDELin: Must have self.matplus and self.Af_dt pre-evaluated at
        t (more efficient in some cases)
        Fill MomentMat (t+1,j)
        from (t,j) and (t+1,j) from (t,j) using forward Euler.
        """
        assert(t>=j)
        M=self.MomentMat
        delta=self.deltaPQMean.view()
        #
        M[j,:,:,t+1,:,:]  = self._contract_tt(M[j,:,:,t,:,:], self.matplus)
        M[j,:,:,t+1,:,:] += self._outerproduct(delta[j,:,:], self.Af_dt)
        # transpose
        M[t+1,:,:,j,:,:]  = self._contract(self.matplus,  M[t,:,:,j,:,:])
        M[t+1,:,:,j,:,:] += self._outerproduct(self.Af_dt, delta[j,:,:])
    #
    def _nondiag_be_update(self,j,t):
        """
        SDELin: Must have self.matminus and self.Ab pre-evaluated at t
        Fill  MomentMat (t-1,j) from (t,j) and
        (t-1,j) from (t,j) using backward Euler.
        """
        assert(t<=j)
        M=self.MomentMat
        delta=self.deltaPQMean.view()
        M[j,:,:,t-1,:,:]  = self._contract_tt(M[j,:,:,t,:,:], self.matminus)
        M[j,:,:,t-1,:,:] += self._outerproduct(delta[j,:,:], self.Ab_dt)
        # transpose
        M[t-1,:,:,j,:,:]  = self._contract(self.matminus, M[t,:,:,j,:,:])
        M[t-1,:,:,j,:,:] += self._outerproduct(self.Ab_dt, delta[j,:,:])
    #
    def _outerproduct(self,A,B):
        return np.einsum('ac,bd->acbd',A,B)
    #
    # this is different to _contract_t in diffeo.py!!
    def _contract_tt(self,A,B):
         return np.tensordot(A,B,axes=([2,3],[2,3]))
    #
    def condition_mat(self,data_var):
        """
        SDELin: Condition the initial and final distributions
        based on the observed data Note that LinPath includes the
        initial and final observations in the first and last
        elements

        """
        print("Computing conditional covariance with data variance: ",
              data_var)
        C12 = self.C[:,:,:,[0,-1],self.d:,:]
        # Assemble C22 carefully to make sure have all the parts
        N = self.d*self.N
        C22_1 = self.C[0, self.d:,:,0, self.d:,:] # No identity part
        C22_2 = self.C[-1,self.d:,:,0, self.d:,:]
        C22_3 = self.C[0, self.d:,:,-1,self.d:,:]
        C22_4 = self.C[-1,self.d:,:,-1,self.d:,:]
        C22 = np.zeros((2,self.d,self.N,2,self.d,self.N))
        C22[0,:,:,0,:,:] = C22_1
        C22[1,:,:,0,:,:] = C22_2
        C22[0,:,:,1,:,:] = C22_3
        C22[1,:,:,1,:,:] = C22_4

        # Add on the identity elements
        for i in range(self.d):
                for j in range(self.N):
                    C22[0,i,j,0,i,j] += data_var
                    C22[1,i,j,1,i,j] += data_var

        # Convert to matrix, compute inversion, reshape to tensor again
        C22r = np.reshape(C22,(2*N,2*N))
        C22inv = np.reshape(LA.inv(C22r),
                            (2,self.d,self.N,
                             2,self.d,self.N))
        #
        dataterm =  -self.deltaPQMean[[-1,0],self.d:,:]
        # Update the distribution using standard formula
        self.CondMean=np.concatenate((self.PLinPath,self.QLinPath),axis=1)\
                       +self.deltaPQMean\
                       +np.tensordot(C12,
                                     np.tensordot(C22inv,dataterm,
                                                  axes=([3,4,5],[0,1,2])),
                                     axes=([3,4,5],[0,1,2]))
        self.CondC=np.copy(self.C)

        self.CondC=self.CondC\
                    -np.tensordot(C12,
                               np.tensordot(C22inv,C12,
                                         axes=([3,4,5],[3,4,5])),
                               axes=([3,4,5],[0,1,2]))
    #
    def sample(self):
        """
        SDELin: Samples from N(CondMean,CondC)
        and stores resulting paths in D
        """
        p=np.prod(self.CondC.shape[0:3])
        CondCMat = np.reshape(self.CondC,(p,p))
        #cholS = spla.cholesky(CondCMat, lower=True)
        #cholS = scipy.linalg.cholesky(Gmat, lower=True)
        #xi = np.random.randn(CondCmat.shape[1],1)
        #out1=dot(cholS, xi)
        out1=np.random.multivariate_normal(np.ravel(self.CondMean), CondCMat)
        self.Samples = np.reshape(out1,self.CondC.shape[0:3])
        self.Ppath=self.Samples[:,:2,:]
        self.Qpath=self.Samples[:,2:,:]
    #
    def get_grid_stats(self,gx,gy,no_samples):
        """
        SDELin:
        """
        m1x=np.zeros_like(gx); m1y=np.zeros_like(gy); m2x=np.zeros_like(m1x); m2y=np.zeros_like(m1y)
        for i in range(no_samples):
            self.sample()
            wgx,wgy=self.diffeo_arrays(gx,gy)
            m1x+=wgx;    m1y+=wgy
            m2x+=wgx**2; m2y+=wgy**2
        m_gx=m1x/no_samples; m_gy=m1y/no_samples
        m2_gx=m2x/no_samples; m2_gy=m2y/no_samples
        factor=no_samples/(no_samples-1)
        vx=(m2_gx-m_gx**2)*factor
        vy=(m2_gy-m_gy**2)*factor
        if np.any(vx+vy<0):
            print("vx+_vy negative ",np.min(vx+vy), ", setting to zero")
        sd=np.sqrt(np.maximum(vx+vy,0.))
        return m_gx,m_gy,sd
    #
    def draw_cond_mean_path(self):
        """
        SDELin: draw conditioned paths (used in run2.py)
        """
        utility.draw_path(self.CondMean[:,2:,:])
    #
    def plot_q(self):
        """
        SDELin:
        used in run2.py
        """
        # Get the diagonal from self.C
        std = np.zeros((self.no_steps+1,self.d,self.N))
        std2 = np.zeros((self.no_steps+1,self.d,self.N))
        d=1
        for i in range(self.d,2*self.d):
                for j in range(self.N):
                    for k in range(self.no_steps+1): #
                        tmp=self.CondC[k,i,j,k,i,j]
                        if tmp>0.:
                            std[k,i-self.d,j] = sqrt(tmp)
                        else:
                            print("\n\nNegative diagonal: ", tmp,"\n\n\n")
                            sys.stdout.flush()
                        std2[k,i-self.d,j] = sqrt(self.C[k,i,j,k,i,j])
        timerange = np.linspace(0,1,num=self.no_steps+1)
        for i in range(self.N):
            cstr=[1., .8, 0.]#yellow
            cstr1=[0., .8, 1.]#blue
            plt.plot(timerange, self.CondMean[:,2+d,i], '-', color=cstr1,
                     linewidth=0.5)
            plt.fill_between(timerange, self.CondMean[:,2+d,i]-std2[:,d,i],
                             self.CondMean[:,2+d,i]+std2[:,d,i],color=cstr,alpha=0.3)
        for i in range(self.N):
            plt.fill_between(timerange,
                             self.CondMean[:,2+d,i]-std[:,d,i],
                             self.CondMean[:,2+d,i]+std[:,d,i],color=cstr,alpha=0.1)
    #
    def _get_cond_sd(self):
        """
        SDELin: return two diagonal vectors, with entry for
        each t and r landmark containing conditional standard
        deviation

        """
        CondCL_RR=self.CondC[0, self.d:, :, 0, self.d:, :]
        CondCL_RT=self.CondC[0, self.d:, :,-1, self.d:, :]
        CondCL_TT=self.CondC[-1,self.d:, :,-1, self.d:, :]
        self.CondCL_RR=CondCL_RR
        self.CondCL_RT=CondCL_RT
        self.CondCL_TT=CondCL_TT
        #
        vt=np.zeros(self.N)
        vr=np.copy(vt)
        vrt=np.copy(vt)
        for i in range(self.N):
            vr[i] =sqrt(LA.norm(self.CondCL_RR[:,i,:,i]))
            vt[i] =sqrt(LA.norm(self.CondCL_TT[:,i,:,i]))
            vrt[i]=sqrt(LA.norm(self.CondCL_RT[:,i,:,i]))
        vtm=max(vt); vrm=max(vr); vrtm=max(vrt);
        mymax=max([vtm,vrm,vrtm])
        vtm=min(vt); vrm=min(vr);
        mymin=min([vtm,vrm])
        return vt,vr,vrt,mymax,mymin
    #
    #
    def sd_plot(self,include_color=True):
        """
        SDELin: add circles on landmarks to indicate standard deviation
        used in run2.py
        """
        Qr=(self.CondMean[0,2:4,:])
        Qt=(self.CondMean[-1,2:4,:])
        QLanPath=np.zeros((self.no_steps+1,self.d,self.N))
        QLanPath[:,0:2,:]=self.QLinPath[:,0:2,:]+self.deltaPQMean[:,2:4,:]
        vt,vr,vrt,mymax,mymin=self._get_cond_sd()
        #fig=plt.figure(5)
        plt.axis('equal')
        # jet = cm = plt.get_cmap('jet')
        cm = mpl.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],
                                      [1., .8, 0.], [1., .4, 0.]])
        cm.set_over((1., 0., 0.))
        cm.set_under((0., 0., 1.))
        bounds = [-1., -.5, 0., .5, 1.]
        norm = mpl.colors.BoundaryNorm(bounds, cm.N)
        cNorm  = colors.Normalize(vmin=mymin, vmax=mymax)
        scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
        utility.plot_reference(Qr,shadow=1)
        utility.plot_target(Qt,shadow=1)
        patches = []
        rmin=0.01
        for i in range(self.N):
            tmp1=vt[i]# target
            if include_color:
                colorVal = scalarMap.to_rgba(tmp1)
            else:
                colorVal=[0.2,0.2,0.2]
            # add a circle
            if tmp1>rmin:
                circle = mpatches.Circle(Qr[:,i], tmp1,facecolor=colorVal,alpha=0.9,edgecolor='none')
                plt.axes().add_patch(circle)


    #        p1=plt.plot(Qt[0,i],Qt[1,i],'*',
    #                    markersize=4,
    #                    markeredgecolor=colorVal,
    #                    color=colorVal)
            tmp2=vr[i]# reference
            if include_color:
                colorVal = scalarMap.to_rgba(tmp2)
            else:
                colorVal=[0.2,0.2,0.2]
            if tmp1>rmin:
                circle = mpatches.Circle(Qt[:,i], tmp1,facecolor=colorVal,alpha=0.9,edgecolor='none')
                plt.axes().add_patch(circle)


    #        p2=plt.plot(Qr[0,i],Qr[1,i],'.',
    #                    markersize=5,color=colorVal)
            if include_color:
                colorVal = scalarMap.to_rgba((tmp1+tmp2)/2)
            else:
                colorVal=[0.5,0.5,0.5]

            plt.plot(QLanPath[:,0,i],
                     QLanPath[:,1,i],linewidth=0.5,alpha=0.9,color=colorVal)
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
        #collection = PatchCollection(patches,lw=0.,facecolors="0.4",alpha=0.7)
        #plt.axes().add_collection(collection)
        scalarMap._A = []
        if include_color:
            plt.colorbar(scalarMap)
    #



#
##############################################
class MAP1(SDE):
    """
    MAP1: Find minimum of log posterior pdf

    beta*H(u0)+0.5*c*( |q0-qr|^2+|q1-qt|^2)

    for c=1/(4*data_var^2) and beta=inverse temperature
    and u0=(p0,q_0) and q1=time-one map of (p_0,q_0) under Ham.
    """
    def __init__(self, G, report_flag=False):
        """
        MAP1:
        """
        SDE.__init__(self, G, report_flag)
        self.TOL=1e-3 # for gradient checker
    #
    def solve(self):
        """
        MAP1: solve optimisation problem
        """
        print("MAP1 solve with",self.no_steps, "steps...")
        sys.stdout.flush()
        if self.report_flag:
            self._gradient_checker(2*self.d*self.N)
        self._init_path()
        uin=np.zeros((2,self.d,self.N))
        uin[1,:,:]=self.landmarks[0,:,:]
        start=timer()
        # exp1: 'CG' 16; 'Powell' 19; Newton-CG 0.6; BFGS 16
        # exp2: 'CG' 59.2; 'Powell' 11.2; 'Newton-CG' 0.5; BFGS 15
        Pout=spo.minimize(self.objective, np.ravel(uin),
                          jac=self.gradient,
                          tol=1e-1,method='Newton-CG',
                          options={'xtol': 5e-3,'disp': 1})
        end=timer()
        print("Run time %3.1f secs" % (end-start))
        #
        u0=Pout['x'].reshape((2,self.d,self.N))
        self.u0=np.ravel(u0)
        self.Prn=u0[0,:,:]
        self.Qrn=u0[1,:,:]
        #
        self.set_path(self.Prn, self.Qrn)
    #
    def objective(self, uin):
        """
        MAP1: define objective function

        evaluate for u0=[p0,q0]

        beta*H(u0)+0.5*c*( |q0-qr|^2+|q1-qt|^2)

        for c=1/(4*data_var^2) and beta=inverse temperature
        and q1=time-1 map of (p_0,q_0) under Ham.
        """
        # define P0, Q0
        uin=uin.reshape((2,self.d,self.N))
        P0=uin[0,:,:]
        Q0=uin[1,:,:]
        # compute P1,Q1 and derivs
        P1,Q1=self.forward(P0,Q0,self.no_steps)
        # evalute H
        H=(self.H(P0,Q0)+self.H(P1,Q1))
        # data term
        d1=np.linalg.norm(Q0-self.landmarks[0,:,:])
        d2=np.linalg.norm(Q1-self.landmarks[1,:,:])
        # evalute obj fun
        return self.half_beta*H  + 0.5*self.c*(d1**2 + d2**2)
    #
    def gradient(self, uin):
        """
        MAP1: compute gradient
        """
        # define P0, Q0
        uin=uin.reshape((2,self.d,self.N))
        P0=uin[0,:,:]
        Q0=uin[1,:,:]
        # define P,Q Jacobian
        DPP, DQP, DPQ, DQQ,P,Q=self._Jac_forward_all(P0, Q0,
                                                     self.no_steps)
        # gradient for H
        H_p=(self.Dp(P0,Q0)
                 +self._contract_t(self.Dp(P,Q),DPP)
                 +self._contract_t(self.Dq(P,Q),DQP))
        H_q=(self.Dq(P0,Q0)
                +self._contract_t(self.Dp(P,Q),DPQ)
                +self._contract_t(self.Dq(P,Q),DQQ))
        # gradient for data
        data_p= self._contract_t((Q-self.landmarks[1,:,:]),DQP)
        data_q= (Q0-self.landmarks[0,:,:])+self._contract_t((Q-self.landmarks[1,:,:]),DQQ)
        # gradient
        gp=self.half_beta*H_p + self.c*data_p
        gq=self.half_beta*H_q + self.c*data_q
        #
        return np.concatenate((gp.flatten(),gq.flatten()))
    #
        #
    def GN_Hessian(self):
        """
        MAP1: An approximation to the Hessian, using
        Gauss-Netwon for the two terms of nonlinear squares.
        #
        """
        # abbreviations :-)
        Ph=self.Prn
        Qh=self.Qrn
        # allocate memory
        Hess_approx=self.c*np.eye(2*self.d*self.N).reshape((2,self.d,self.N,
                                                            2,self.d,self.N))
        # GN Hessian for data terms
        X= self._Hess_data_GN(Ph,Qh,self.no_steps) # target
        self._add_to(Hess_approx,X,[0,1])
        # exact Hessian for H
        Hess_H=self.beta*self.Hessian(Ph,Qh)
        self._add_to(Hess_approx,Hess_H,[0,1])
        return Hess_approx.reshape((2*self.l1,2*self.l1))
    #
    def cov(self):
        """
        MAP1: get covariance from Hessian
        Sum along spatial dimensions and two momenta
        return qq and pp covariance of size N squared
        """
        # print("Compute inverse of H for covariance...")
        sys.stdout.flush()
        H=self.GN_Hessian()
        start=timer()
        C=utility.nearPSD_inv(H,1e-7)
        C=C.reshape((2,self.d,self.N,2,self.d,self.N))
        Cps =(C[0,0,:,0,0,:]+C[0,1,:,0,1,:])/2
        Cqs =(C[1,0,:,1,0,:]+C[1,1,:,1,1,:])/2
        #
        # print("Run time %3.1f secs" % (timer()-start))
        sys.stdout.flush()
        return Cqs,Cps
#
#
##################################################
class MAP2(SDE):
    """
    This is a variant of MAP1 that allow multiple shooting
    """
    def __init__(self, G, report_flag=False):
        """
        MAP2:
        """
        SDE.__init__(self, G, report_flag)
        self.scale_p=1e4
        self.scale_const_q=1e4
        print("MAP2: scaling for momenta ", self.scale_p)
        print("No Jacobian checker :-(")
    #
    def solve(self):
        """
        MAP2: compute initial momentum to satisfy BVP
        by solving nonlinear system
        """
        print("MAP2 solve with steps",
              self.m_no_steps, " and total of ",self.no_steps, " steps...")
        sys.stdout.flush()
        maxiter=1e3
        if self.report_flag:
            self._gradient_checker()
            print("need to write a gradient checker :-(")
            #
        self._init_path()
        noVars=2*(self.m_no_steps.size+1) #[pi,q1 for i=0,..,no_steps]
        # set-up the initial guess
        u0=np.zeros((noVars,self.d, self.N))
        if self.initialguess==1:
            # case 1
            print("Initial guess: flow")
            sys.stdout.flush()
            Q=np.copy(self.landmarks[0,:,:])
            P=np.zeros_like(Q)
            u0[1,:,:]=Q
            for i in range(self.m_no_steps.size):
                P,Q=self.Forward(P,Q,self.m_no_steps[i])
                u0[2*i+3,:,:] =Q
                u0[2*i+2  ,:,:] =P/self.scale_p
        elif self.initialguess==0:
            print("Initial guess: linear in q")
            sys.stdout.flush()
            u0[1,:,:]=self.landmarks[0,:,:]
            j=0;
            delta=(self.landmarks[1,:,:]-self.landmarks[0,:,:])
            for i in range(0,self.m_no_steps.size):
                j=j+self.m_no_steps[i]
                u0[2*i+3,:,:]=self.landmarks[0,:,:]+delta*(j/self.no_steps)
        # allocate memory for return
        uout=np.zeros(self.d * self.N * noVars)
        # optimize
        start=timer()
        cons = ({'type': 'eq',
                 'fun': self.constraint,
                 'jac': self.constraint_gradient
                 })
        uout=spo.minimize(self.objective,  u0,    jac=self.gradient,
                          method='SLSQP',
                          constraints=cons,
                          options={'ftol': 1e-6,'disp': 1,'maxiter': maxiter}) #
        end=timer()
        print("Run time %3.1f secs"% (end-start))
        print(uout['message'])
        if uout['success']:
            X=np.array(uout['x']).reshape(noVars, self.d, self.N)
            self.u0=np.ravel(X)
            self.Prn=X[0,:,:]*self.scale_p
            self.Qrn=X[1,:,:]
        else:
            print("it didn't work.")
            assert False
        #
        self.set_path(self.Prn, self.Qrn)
    #
    def objective(self, uin):
        """
        MAP2: evaluate for u0=[pi,qi i=0,...,M]
        M=2 (standard shooting)
        M=2*self.m_no_steps.size

        0.5*beta*(H(p1,q1)+H(pM,qM))+c*( |q1-qr|^2+|q1-qt|^2))

        for c=1/(2*data_var^2) and beta=inverse temperature
        """
        noVar=2*(self.m_no_steps.size+1)
        uin=uin.reshape((noVar,self.d,self.N))
        # evaluate H
        H=(   self.H(uin[0 ,:,:]*self.scale_p, uin[ 1,:,:])
            + self.H(uin[-2,:,:]*self.scale_p, uin[-1,:,:]) )
        # data term
        n1=np.linalg.norm(uin[ 1,:,:]-self.landmarks[0,:,:])
        n2=np.linalg.norm(uin[-1,:,:]-self.landmarks[1,:,:])
        # evalute obj fun
        return self.half_beta * H + self.c * (n1**2 + n2**2)
    #
    def gradient(self, uin):
        """
        MAP2:
        """
        noVar=2*(self.m_no_steps.size+1)
        uin=uin.reshape((noVar, self.d, self.N))
        out=np.zeros_like(uin)
        # gradient for H
        out[ 0,:,:] =self.scale_p*self.Dp(uin[ 0,:,:]*self.scale_p, uin[ 1,:,:])*self.half_beta
        out[ 1,:,:] =             self.Dq(uin[ 0,:,:]*self.scale_p, uin[ 1,:,:])*self.half_beta
        out[-2,:,:] =self.scale_p*self.Dp(uin[-2,:,:]*self.scale_p, uin[-1,:,:])*self.half_beta
        out[-1,:,:] =             self.Dq(uin[-2,:,:]*self.scale_p, uin[-1,:,:])*self.half_beta
        # gradient for data
        out[ 1,:,:] +=2*(uin[ 1,:,:]-self.landmarks[0,:,:])*self.c
        out[-1,:,:] +=2*(uin[-1,:,:]-self.landmarks[1,:,:])*self.c
        # gradient
        return np.ravel(out)
    #
    def constraint(self, uin):
        """
        MAP2
        """
        # reshape current state
        noVar =2*(self.m_no_steps.size+1)
        noCons=2*(self.m_no_steps.size)
        uin=uin.reshape((noVar, self.d, self.N))
        out=np.zeros((noCons,self.d,self.N))
        for i in range(self.m_no_steps.size):
            j=2*i
            P=uin[j,  :,:]*self.scale_p
            Q=uin[j+1,:,:]
            P,Q=self.forward(P,Q,self.m_no_steps[i])
            out[j  ,:,:] = P/self.scale_p - uin[j+2,:,:]
            out[j+1,:,:] = ( Q - uin[j+3,:,:] ) / self.scale_const_q
        return np.ravel(out)
    #
    def constraint_gradient(self, uin):
        """
        MAP2:
        """
        # reshape current state
        noVars=2*(self.m_no_steps.size+1)
        noCons=2*(self.m_no_steps.size)
        uin=uin.reshape((noVars,self.d,self.N))
        # define usefully sized identity matrix
        id=np.eye(self.l1).reshape(self.s1)
        # allocate memory for return
        bigJ=np.zeros((noCons,self.d,self.N,
                        noVars,self.d,self.N))
        # intermediate shoots
        for i in range(self.m_no_steps.size):
            j=2*i
            P=np.copy(uin[j,  :,:]) * self.scale_p
            Q=np.copy(uin[j+1,:,:])
            DPP,DQP,DPQ,DQQ=self._Jac_forward_both(P,Q,self.m_no_steps[i])
            # P/self.scale_p -uin[j+2,:,:]
            bigJ[j,:,:,j,  :,:] =DPP # dp/dp
            bigJ[j,:,:,j+1,:,:] =DPQ/self.scale_p   # dp/dq
            bigJ[j,:,:,j+2,:,:] =-id #
            # (Q-uin[j+3,:,:])/self.scale_const_q
            bigJ[j+1,:,:,j,:,:]  =DQP*self.scale_p/self.scale_const_q        # dq/dp
            bigJ[j+1,:,:,j+1,:,:]=DQQ/self.scale_const_q   # dq/dq
            bigJ[j+1,:,:,j+3,:,:]=-id/self.scale_const_q
        #
        bigJ=bigJ.reshape((self.l1*noCons,self.l1*noVars))
        return bigJ
    #
    def GN_Hessian(self):
        """
        MAP2: An approximation to the Hessian, using
        Gauss-Netwon for the two terms of nonlinear squares.
        #
        """
        # abbreviations :-)
        Ph=self.Prn
        Qh=self.Qrn
        # allocate memory
        Hess_approx=self.c*np.eye(2*self.d*self.N).reshape((2,self.d,self.N,
                                                            2,self.d,self.N))
        # GN Hessian for data terms
           # GN Hessian for data terms
        X= self._Hess_data_GN(Ph,Qh,self.no_steps) # target
        self._add_to(Hess_approx,X,[0,1])
        # exact Hessian for H
        Hess_H=self.beta*self.Hessian(Ph,Qh)
        self._add_to(Hess_approx,Hess_H,[0,1])
        return Hess_approx.reshape((2*self.l1,2*self.l1))
    #
    def cov(self):
        """
        MAP2: get covariance from Hessian
        Sum along spatial dimensions and two momenta
        return qq and pp covariance of size N squared
        """
        # print("Compute inverse of H for covariance...")
        sys.stdout.flush()
        start=timer()
        H=self.GN_Hessian()
        C=utility.nearPSD_inv(H,1e-7)
        print(H.shape,C.shape,self.d,self.N)
        C=C.reshape((2,self.d,self.N,2,self.d,self.N))
        Cps =(C[0,0,:,0,0,:]+C[0,1,:,0,1,:])/2
        Cqs =(C[1,0,:,1,0,:]+C[1,1,:,1,1,:])/2
        #
        # print("Run time %3.1f secs" % (timer()-start))
        sys.stdout.flush()
        return Cqs,Cps
##############################################
class MAP3(SDE):
    """
    MAP3: MAP-type optimisation to find diffeo
    for optimising

    beta*H(p_half,q_half)
    +0.5*(|tp_half-e^(-G(q_half) lam/2) p_half|^2 / sig^2)
    +0.5*c*( |q0-qr|^2+|q1-qt|^2)

    for q0 =time-1/2 map from (q_half,p_half)
    q1 =time-1/2 map from (q_half,tp_half)
    c=1/(4*data_var^2)
    beta=inverse temperature

    Note: MAP3 and MAP4 both do two images, but slightly different objective
    functions (only two momentum in MAP3, three momentum in MAP4)

    """
    def __init__(self, G, report_flag=False):
        """
        MAP3: init
        """
        SDE.__init__(self, G, report_flag)
    #
    def solve(self):
        """
        MAP3: solve optimisation problem
        """
        print("MAP3 solve with",self.no_steps, "steps...")
        sys.stdout.flush()
        if self.report_flag:
            self._gradient_checker(3*self.d*self.N)
        self._init_path()
        uin=np.zeros((3,self.d,self.N))
        uin[1,:,:]=self.landmarks[0,:,:]
        start=timer()
        Pout=spo.minimize(self.objective, np.ravel(uin),
                          jac=self.gradient,
                          tol=1e-8,method='CG',
                          options={'gtol': 1e-8,'maxiter':int(1e4),'disp': 1})
        end=timer()
        print("Run time %3.1f secs" % (end-start))
        #print(Pout['message'])
        u0=Pout['x']
        #
        self.u0=u0
        u0=u0.reshape((3,self.d,self.N))
        self.Ph=u0[0,:,:]
        self.Qh=u0[1,:,:]
        self.tPh=u0[2,:,:]
    #
    def objective(self, uin):
        """
        MAP3: define objective function
        uin=[Phalf, Qhalf, tPhalf]
        """
        # extract data
        uin=uin.reshape((3,self.d,self.N))
        Ph=uin[0,:,:]
        Qh=uin[1,:,:]
        tPh=uin[2,:,:]
        # evalute H
        H=self.H(Ph,Qh)
        # evalute OU term
        factor=1/(self.sig**2)
        OUterm=factor*self.OU_fun(Ph,Qh,tPh,self.lam)
        # compute data terms
        half_no_steps=self.no_steps//2
        P0,Q0=self.forward(Ph,Qh,half_no_steps)
        n1=np.linalg.norm(Q0-self.landmarks[0,:,:])**2
        P1,Q1=self.forward(tPh,Qh,half_no_steps)
        n2=np.linalg.norm(Q1-self.landmarks[1,:,:])**2
        # evalute obj fun
        return self.beta*H + OUterm + 0.5*self.c*(n1+n2)
    #
    def gradient(self, uin):
        """
        MAP3: compute gradient
        """
        # extract data from uin
        uin=uin.reshape((3,self.d,self.N))
        Ph=uin[0,:,:]
        Qh=uin[1,:,:]
        tPh=uin[2,:,:]
        # allocate memory for data
        g=np.zeros((3,self.d,self.N))
        # gradient for Hamiltonian
        grad_H=self.beta*self.gradH(Ph,Qh)
        g[[0,1],:,:]+=grad_H
        # gradient for OU term
        grad_OU=self.grad_OU(Ph,Qh,tPh,self.lam)
        g[[0,1,2],:,:]+= (1/self.sig**2)*grad_OU
        # gradient for data terms
        half_no_steps=self.no_steps//2
        grad_data0=self._grad_data(Ph,Qh,self.landmarks[0,:,:],half_no_steps)
        g[[0,1],:,:]+=grad_data0
        grad_data1=self._grad_data(tPh,Qh,self.landmarks[1,:,:],half_no_steps)
        g[[2,1],:,:]+=grad_data1
        return g.ravel()
        #
    #
    #
    def GN_Hessian(self):
        """
        MAP3: An approximation to the Hessian, using
        Gauss-Netwon for the two terms of nonlinear squares.
        """
        # abbreviations :-)
        Ph=self.Ph
        Qh=self.Qh
        tPh=self.tPh
        # allocate memory
        Hess_approx=np.zeros((3,self.d,self.N,3,self.d,self.N))
        # GN Hessian for data terms
        half_no_steps=self.no_steps//2
        X= self._Hess_data_GN(Ph,Qh,half_no_steps) # reference
        self._add_to(Hess_approx,X,[0,1])
        X= self._Hess_data_GN(tPh,Qh,half_no_steps) # target
        # note the indices are inverted here as the data is ordered
        # as Ph,Qh,tPh
        self._add_to(Hess_approx,X,[2,1])
        # GN Hessian for OU
        X=self.Hess_OU_GN(Ph,Qh,tPh,self.lam)/self.sig**2
        self._add_to(Hess_approx,X,[0,1,2])
        # exact Hessian for H
        Hess_H=self.beta*self.Hessian(Ph,Qh)
        self._add_to(Hess_approx,Hess_H,[0,1])
        return Hess_approx.reshape((3*self.l1,3*self.l1))
    #
    def cov(self):
        """
        MAP3: get covariance from Hessian
        Sum along spatial dimensions and two momenta
        return qq and pp covariance of size N squared
        """
        # print("Compute inverse of H for covariance...")
        sys.stdout.flush()
        start=timer()
        H=self.GN_Hessian()
        C=utility.nearPSD_inv(H,1e-7)
        C=C.reshape((2,self.d,self.N,2,self.d,self.N))
        Cps0=(C[0,0,:,0,0,:]+C[0,1,:,0,1,:])/2
        Cqs =(C[1,0,:,1,0,:]+C[1,1,:,1,1,:])/2
        #
        Cps=(Cps0+Cps1)/2
        # print("Run time %3.1f secs" % (timer()-start))
        sys.stdout.flush()
        return Cqs,Cps
    #
    # def cov2(self,H):
    #     """
    #     MAP3: get covariance from Hessian
    #     Sum along spatial dimensions and two momenta
    #     return qq and pp covariance of size N squared
    #     """
    #     print("Compute inverse of H for covariance...")
    #     sys.stdout.flush()
    #     start=timer()
    #     Hq=H[self.l1:2*self.l1,self.l1:2*self.l1]
    #     C=utility.nearPSD_inv(Hq)
    #     C=C.reshape((self.d,self.N,self.d,self.N))
    #     Cqs=(C[0,:,0,:]+C[1,:,1,:])/2
    #     print("diag Cqs",np.diag(Cqs))
        #     return Cqs
#
#
#
##############################################
class MAP4(SDE):
    """
    MAP4: MAP-type optimisation based on MAP3 with multiple landmarks

    beta*H(p,q)
    +0.5*(|p-e^(-G(q) lam/2) p_j|^2 )/ sig^2)
    +0.5*c* sum_j ( |time-half(q_j,p_j) - qdata_j|^2)

    for time-half map is Hamiltonian flow
    c=1/(4*data_var^2)
    beta=inverse temperature

    """
    def __init__(self, G, report_flag=False):
        """
        MAP4: init
        """
        SDE.__init__(self, G, report_flag)
    #
    def solve(self):
        """
        MAP4: solve optimisation problem
        """
        print("MAP4 solve with",self.no_steps, "steps...")
        sys.stdout.flush()
        no_lm=self.landmarks.shape[0]
        if self.report_flag:
            self._gradient_checker((2+no_lm)*self.d*self.N)
        self._init_path()
        uin=np.zeros((no_lm+2,self.d,self.N))
        uin[1,:,:]=self.landmarks[0,:,:]
        start=timer()
        # CG 7; Powell; BFGS 20; Newton-CG 2.5
        Pout=spo.minimize(self.objective, np.ravel(uin),
                          jac=self.gradient,
                          tol=1e-1,method='Newton-CG',
                          options={'xtol': 5e-3,'disp': 1})
        end=timer()
        print("Run time %3.1f secs" % (end-start))
        #print(Pout['message'])
        u0=Pout['x']
        #
        self.u0=u0
        u0=u0.reshape((no_lm+2,self.d,self.N))
        self.Ph=u0[0,:,:]
        self.Qh=u0[1,:,:]
        self.tPh=u0[2,:,:]
    #
    def objective(self, uin):
        """
        MAP4: define objective function
        uin=[P, Q, Pj] for j=1,..., number of landmark sets
        """
        # extract data
        no_lm=self.landmarks.shape[0]
        uin=uin.reshape((2+no_lm,self.d,self.N))
        Ph=uin[0,:,:]
        Qh=uin[1,:,:]
        # evalute H
        H=self.H(Ph,Qh)
        factor=1/(self.sig**2)
        half_no_steps=self.no_steps//2
        n=0
        OUterm=0
        self.pre_comp_expG2(Qh,self.lam)
        for i in range(no_lm):
            tPh=uin[2+i]
            OUterm+=self.OU_fun(tPh,Qh,Ph,self.lam)
            P0,Q0=self.forward(tPh,Qh,half_no_steps)
            n+=np.linalg.norm(Q0-self.landmarks[i,:,:])**2
        # evalute obj fun
        return self.beta*H + factor*OUterm + 0.5*self.c*n
    #
    def gradient(self, uin):
        """
        MAP4: compute gradient
        """
        # extract data from uin
        no_lm=self.landmarks.shape[0]
        uin=uin.reshape((2+no_lm,self.d,self.N))
        Ph=uin[0,:,:]
        Qh=uin[1,:,:]
        # allocate memory for data
        g=np.zeros_like(uin)
        # gradient for Hamiltonian
        grad_H=self.beta*self.gradH(Ph,Qh)
        g[[0,1],:,:]+=grad_H
        half_no_steps=self.no_steps//2
        self.pre_comp_dexpG2(Qh,self.lam)
        for i in range(no_lm):
            tPh=uin[2+i]
            # gradient for OU term
            grad_OU=self.grad_OU(tPh,Qh,Ph,self.lam)
            g[[2+i,1,0],:,:]+= (1/self.sig**2)*grad_OU
            grad_data=self._grad_data(tPh,Qh,
                                      self.landmarks[i,:,:],half_no_steps)
            g[[2+i,1],:,:]+=grad_data
        return g.ravel()
        #
    #
    def _add_to(self,A,B,ind):
        """
        MAP4: for 6-index tensors A and B,
        add entries from B to specified entries of A. Typically, B
        is smaller than A and this cannot be achieved directly
        """
        for i in range(len(ind)):
            A[ind,:,:,ind[i],:,:]+= B[:,:,:,i,:,:]
    #
    def GN_Hessian(self):
        """
        MAP4: An approximation to the Hessian, using
        Gauss-Netwon for the two terms of nonlinear squares.
        """
        # abbreviations :-)
        no_lm=self.landmarks.shape[0]
        Ph=self.Ph
        Qh=self.Qh
        uin=self.u0.reshape((2+no_lm,self.d,self.N))
        # allocate memory
        Hess_approx=np.zeros((2+no_lm,self.d,self.N,
                              2+no_lm,self.d,self.N))
        # GN Hessian for data terms
        half_no_steps=self.no_steps//2
        for i in range(no_lm):
            tPh=uin[2+i,:,:]
            X= self._Hess_data_GN(tPh,Qh,half_no_steps) # reference
            self._add_to(Hess_approx,X,[2+i,1])
            X=self.Hess_OU_GN(tPh,Qh,Ph,self.lam)/self.sig**2
            self._add_to(Hess_approx,X,[2+i,1,0])
        # exact Hessian for H
        Hess_H=self.beta*self.Hessian(Ph,Qh)
        self._add_to(Hess_approx,Hess_H,[0,1])
        return Hess_approx.reshape(((2+no_lm)*self.l1,(2+no_lm)*self.l1))
    #
    def cov(self):
        """
        MAP4: get covariance from Hessian
        Sum along spatial dimensions and two momenta
        return qq and pp covariance of size N squared
        """
        # print("Compute inverse of H for covariance...")
        sys.stdout.flush()
        H=self.GN_Hessian()
        start=timer()
        C=utility.nearPSD_inv(H,1e-7)
        no_lm=self.landmarks.shape[0]
        C=C.reshape((2+no_lm,self.d,self.N,2+no_lm,self.d,self.N))
        Cqs =(C[1,0,:,1,0,:]+C[1,1,:,1,1,:])/2
        # print("Run time %3.1f secs" % (timer()-start))
        sys.stdout.flush()
        return Cqs,0
    #
#
# #
# #
# ##############################################
# class MAP5(SDE):
#     """
#     MAP5: MAP-type optimisation based on MAP3 with multiple landmarks
#
#     beta*H(p,q)
#     +0.5*(|p-e^(-G(q) lam/2) p_j|^2 )/ sig^2)
#     +0.5*c* sum_j ( |time-half(q_j,p_j) - qdata_j|^2)
#
#     for time-half map is Hamiltonian flow
#     c=1/(4*data_var^2)
#     beta=inverse temperature
#
#     """
#     def __init__(self, G, report_flag=False):
#         """
#         MAP5: init
#         """
#         SDE.__init__(self, G, report_flag)
#     #
#     def solve(self):
#         """
#         MAP5: solve optimisation problem
#         """
#         print("MAP5 solve with",self.no_steps, "steps...")
#         sys.stdout.flush()
#         no_lm=self.landmarks.shape[0]
#         if self.report_flag:
#             self._gradient_checker((2+no_lm)*self.d*self.N)
#         self.init_path()
#         uin=np.zeros((no_lm+2,self.d,self.N))
#         uin[1,:,:]=self.landmarks[0,:,:]
#         start=timer()
#         Pout=spo.minimize(self.objective, np.ravel(uin),
#                           jac=self.gradient,
#                           tol=1e-1,method='CG',
#                           options={'gtol': 1e-1,'disp': 1})
#         end=timer()
#         print("Run time %3.1f secs" % (end-start))
#         #print(Pout['message'])
#         u0=Pout['x']
#         #
#         self.u0=u0
#         u0=u0.reshape((no_lm+2,self.d,self.N))
#         self.Ph=u0[0,:,:]
#         self.Qh=u0[1,:,:]
#         P1,Q1=self.forward(self.Ph,self.Qh,self.no_steps)
#         self.Qav=Q1
#         self.tPh=u0[2,:,:]
#     #
#     def objective(self, uin):
#         """
#         MAP5: define objective function
#         uin=[P, Q, Pj] for j=1,..., number of landmark sets
#         """
#         # extract data
#         no_lm=self.landmarks.shape[0]
#         uin=uin.reshape((2+no_lm,self.d,self.N))
#         Ph=uin[0,:,:]
#         Qh=uin[1,:,:]
#         # evalute H
#         H=self.H(Ph,Qh)
#         factor=1/(self.sig**2)
#         half_no_steps=self.no_steps//2
#         n=0
#         OUterm=0
#         self.pre_comp_expG2(Qh,self.lam)
#         for i in range(no_lm):
#             tPh=uin[2+i]
#             OUterm+=self.OU_fun(Ph,Qh,tPh,self.lam)
#             P0,Q0=self.forward(tPh,Qh,self.no_steps)
#             n+=np.linalg.norm(Q0-self.landmarks[i,:,:])**2
#         # evalute obj fun
#         return self.beta*H + factor*OUterm + 0.5*self.c*n
#     #
#     def gradient(self, uin):
#         """
#         MAP5: compute gradient
#         """
#         # extract data from uin
#         no_lm=self.landmarks.shape[0]
#         uin=uin.reshape((2+no_lm,self.d,self.N))
#         Ph=uin[0,:,:]
#         Qh=uin[1,:,:]
#         # allocate memory for data
#         g=np.zeros_like(uin)
#         # gradient for Hamiltonian
#         grad_H=self.beta*self.gradH(Ph,Qh)
#         g[[0,1],:,:]+=grad_H
#         self.pre_comp_dexpG2(Qh,self.lam)
#         for i in range(no_lm):
#             tPh=uin[2+i]
#             # gradient for OU term
#             grad_OU=self.grad_OU(Ph,Qh,tPh,self.lam)
#             g[[0,1,2+i],:,:]+= (1/self.sig**2)*grad_OU
#             grad_data=self._grad_data(tPh,Qh,
#                                       self.landmarks[i,:,:],self.no_steps)
#             g[[2+i,1],:,:]+=grad_data
#         return g.ravel()
#         #
#     #
#     def _add_to(self,A,B,ind):
#         """
#         MAP5: for 6-index tensors A and B,
#         add entries from B to specified entries of A. Typically, B
#         is smaller than A and this cannot be achieved directly
#         """
#         for i in range(len(ind)):
#             A[ind,:,:,ind[i],:,:]+= B[:,:,:,i,:,:]
#     #
#     def GN_Hessian(self):
#         """
#         MAP5: An approximation to the Hessian, using
#         Gauss-Netwon for the two terms of nonlinear squares.
#         """
#         # abbreviations :-)
#         no_lm=self.landmarks.shape[0]
#         Ph=self.Ph
#         Qh=self.Qh
#         uin=self.u0.reshape((2+no_lm,self.d,self.N))
#         # allocate memory
#         Hess_approx=np.zeros((2+no_lm,self.d,self.N,
#                               2+no_lm,self.d,self.N))
#         # GN Hessian for data terms
#         for i in range(no_lm):
#             tPh=uin[2+i,:,:]
#             X= self._Hess_data_GN(tPh,Qh,self.no_steps) # reference
#             self._add_to(Hess_approx,X,[2+i,1])
#             X=self.Hess_OU_GN(Ph,Qh,tPh,self.lam)/self.sig**2
#             self._add_to(Hess_approx,X,[1,2+i,0])
#         # exact Hessian for H
#         Hess_H=self.beta*self.Hessian(Ph,Qh)
#         self._add_to(Hess_approx,Hess_H,[0,1])
#         return Hess_approx.reshape(((2+no_lm)*self.l1,(2+no_lm)*self.l1))
#     #
#     def cov(self,H):
#         """
#         MAP5: get covariance from Hessian
#         Sum along spatial dimensions and two momenta
#         return qq and pp covariance of size N squared
#         """
#         # print("Compute inverse of H for covariance...")
#         sys.stdout.flush()
#         start=timer()
#         C=utility.nearPSD_inv(H,1e-7)
#         no_lm=self.landmarks.shape[0]
#         C=C.reshape((2+no_lm,self.d,self.N,2+no_lm,self.d,self.N))
#         Cqs =(C[1,0,:,1,0,:]+C[1,1,:,1,1,:])/2
#         # print("Run time %3.1f secs" % (timer()-start))
#         sys.stdout.flush()
#         return Cqs,0
#     #
