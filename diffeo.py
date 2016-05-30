from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
exec(open("ground.py").read())
"""
diffeo.py
--------------
For deterministic landmark image registration.

Classes:
Diffeo, Shoot, MultiShoot
"""
#
from timeit import default_timer as timer
#
import scipy.optimize as spo
#
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
#
import skimage
from skimage.transform import warp
#
# mine
import hamiltonian
from hamiltonian import Hamiltonian
import utility
#
"""
Class:  Diffeo(G)
Diffeomorphisms based on Green's fn G (see hamiltonian.py)
"""
class Diffeo(Hamiltonian):
    def __init__(self, G, report_flag=False):
        """
        Diffeo: report_flag=False (quiet) or True (gives progress reports)
        """
        Hamiltonian.__init__(self, G, report_flag)
        # initialize
        N=1 # with one landmark
        self.set_landmarks(np.zeros((2,self.d, N))) # landmarks are indexed [i,j,k]
        # i=landmark set index, j=dimension index, k=landmark index
        #
        self.set_no_steps(5) # no. time steps for discretising ODEs
    #
    def set_landmarks(self, Q):
        """
        Diffeo: set landmarks Q[i,j,k]
        i=landmark set index (i=0 is reference),
        j=dimension index, k=landmark index
        """
        assert Q.shape[0]>=2 # must have a reference and target
        self.N=Q.shape[2]    # number of landmarks
        #
        if self.report_flag:
            print("Set landmarks with size", Q.shape)
        self.landmarks=np.copy(Q)
        # useful shapes
        self.l1=self.d*self.N
        self.s0=(self.d,self.N)
        self.s1=(self.d,self.N,self.d,self.N)
    #
    def copy(self, D):
        """
        Diffeo: copy data from another diffeo object
        """
        self.set_landmarks(D.landmarks)
        self.Qpath=np.copy(D.Qpath)
        self.Ppath=np.copy(D.Ppath)
        self.set_no_steps(D.no_steps)
    #
    def _init_path(self):
        """
        Diffeo: initialise memory for the paths
        """
        self.Qpath=np.zeros((self.no_steps+1,self.d, self.N))
        self.Ppath=np.zeros_like(self.Qpath)
    #
    def set_path(self, Pin, Qin):
        """
        Diffeo: Compute paths by timestepping and store paths for
        position, momentum in Qpath, Ppath [i,j,k]
        i=time index, j= spatial dimension, k=particle number
        """
        assert Qin.shape[0]==self.d
        assert Qin.shape[1]==self.N
        assert Pin.shape==Qin.shape
        #
        Q=np.copy(Qin)
        P=np.copy(Pin)
        #
        self._init_path()
        self.Qpath[0,:,:]=Q
        self.Ppath[0,:,:]=P
        #
        for i in range(self.no_steps):
            P, Q=self._one_step(P, Q)
            self.Qpath[i+1:,:]=Q
            self.Ppath[i+1:,:]=P
    #
    def plot_qpath_01(self, shadow=6):
        """
        Diffeo: plot initial and final landmarks (line and marker)
        """
        utility.plot_RT(self.Qpath[[0,-1],:,:], shadow)
    #
    def plot_warp(self,shadow=0):
        """
        Diffeo: plot warp (intermediate shapes
        and lines connecting landmarks)
        """
        utility.plot_path(self.Qpath,shadow)
    #
    def _diffeo(self, Qin, transpose=False):
        """
        Diffeo: evaluate diffeomoprhism on Qin, using paths
        from set_path

        Better to use diffeo_arrays
        """
        if transpose==False:
            Q=np.copy(Qin)
        else:
            Q=np.copy(Qin.T)
        if Q.ndim==1:
            Q=Q.reshape([Q.shape[0],1])
        for i in range(self.no_steps):
            GQQ=self.G.G2(self.Qpath[i,:,:], Q)
            force=np.tensordot(self.Ppath[i,:,:],GQQ,axes=((1),(0)))
            Q=Q+force*self.dt
        if transpose==True:
            Q=Q.T
        return Q
    #
    def _diffeo_image(self, Qin, s):
        """
        Diffeo: co-ordinates for images are provided as Nx2 matrix.
        Beware: diffeo algorithm works on 2xN matrix and
        needs a rescaled image.
        TODO: adjust scaling and introducing centering
        """
        n=Qin.shape[0]
        # need to switch axis (from matrix ordering to (x,y) ordering)
        ix=0; iy=int(1-ix)
        x=Qin[:,ix]; y=Qin[:,iy]
        xl=s[iy]*0.5; yl=s[ix]*0.5
        # rescale onto [-1,1]^2
        Q=np.vstack( [ (x-xl) / xl, (y-yl) / yl ])
        # apply diffeo
        Q1=self._diffeo(Q)
        # switch back to matrix form
        nx=Q1[ix,:].reshape((n,1))
        ny=Q1[iy,:].reshape((n,1))
        xl=s[1]*0.5; yl=s[0]*0.5
        # rescale back
        Q=np.hstack([xl+nx*xl, yl*ny+yl])
        return Q
    #
    def warp(self, image):
        """
        Diffeo: Warp an image; e.g., as provided by
        import skimage; from skimage import data; image = data.coffee()
        """
        print("Warping image...")
        sys.stdout.flush()
        start=timer()
        t=lambda x: self._diffeo_image(x, image.shape)
        new_image=skimage.transform.warp(image, t)
        print("Time to warp image %3.1f secs" % (timer()-start))
        return new_image
    #
    def diffeo_arrays(self, px, py):
        """
        Diffeo: apply diffeo to nd arrays of x and y co-ordinates.
        For 2d arrays, diffeo is applied row by row.
        Otherwise, flattened and done together.
        """
        s=px.shape
        if px.ndim==1 or px.ndim>2:
            gx=px.ravel();  gy=py.ravel()
            gx,gy=self._diffeo(np.vstack([px,py]))
            gx=gx.reshape(s)
            gy=gy.reshape(s)
        else:
            gx=np.empty_like(px)
            gy=np.empty_like(py)
            for i in range(s[0]):
                out=self._diffeo(np.vstack([px[i,:],py[i,:]]))
                gx[i,:]=out[0,:]
                gy[i,:]=out[1,:]
        return gx,gy
    #
    def plot_warped_grid(self, nl):
        """
        Diffeo: Qpath (start and end) is used to determine
        appropriate x-y ranges.
        nl=number of lines
        """
        g0=self.Qpath[0,:,:]
        g1=self.Qpath[-1,:,:]
        gx,gy,xc,yc=utility.get_grid(g0,g1,nl)
        wgx,wgy=self.diffeo_arrays(gx,gy)
        utility.plot_grid(wgx,wgy)
        plt.xlim(xc)
        plt.ylim(yc)
    #
    def forward(self, P, Q, no_steps):
        """
        Diffeo: apply no_steps to initial P,Q
        """
        for i in range(no_steps):
            P, Q=self._one_step(P, Q)
        return P,Q
    #
    def set_no_steps(self, no_steps):
        """
        Diffeo: sets the multiple time stepping via a two vector
        first entry=no steps per shoot
        second entry=no multiple shoots
        """
        if isinstance(no_steps, list):
            nStepsPerShoot=no_steps[0]
            nShoot=no_steps[1]
            mNoSteps=nStepsPerShoot*np.ones(nShoot,dtype=int)
            self._set_multiple_steps(mNoSteps)
        elif isinstance(no_steps,int):
            self.no_steps=no_steps
            self.dt = 1.0/(no_steps-1.0)
        else:
            assert False
    #
    def _set_multiple_steps(self, stepsin):
        """
        Diffeo: stepsin is a vector of natural numbers,
        indicating number of time steps
        taken on each sub-interval.
        """
        no_steps=int( sum(stepsin))
        self.set_no_steps(no_steps)
        self.m_no_steps=stepsin
        self.initialguess=2 # (0 for linear; 1 for flow; 2 for all zero)
        if self.report_flag:
            self._Jacobian_checker()
    #
    def _one_step(self, P, Q):
        """
        Diffeo:
        Return Euler-step updated P,Q
        """
        HP=self.Dp(P, Q)
        HQ=self.Dq(P, Q)
        Pn=P-HQ*self.dt
        Qn=Q+HP*self.dt
        return Pn, Qn
    #
    def _one_step_Jac_both(self, P, Q, DPP, DQP, DPQ, DQQ):
        """
        Diffeo: Apply one step for Jacobian
        """
        HPP=self.Dpp(P, Q)
        HPQ=self.Dpq(P, Q)
        HQQ=self.Dqq(P, Q)
        # P derivs
        DPPn=DPP-(self._contract_t(HPQ,DPP) + self._contract(HQQ,DQP))*self.dt
        DQPn=DQP+(self._contract(HPP,DPP) + self._contract(HPQ,DQP))*self.dt
        # Q derivs
        DPQn=DPQ-(self._contract_t(HPQ,DPQ) + self._contract(HQQ,DQQ))*self.dt
        DQQn=DQQ+(self._contract(HPP,DPQ) + self._contract(HPQ,DQQ))*self.dt
        return DPPn, DQPn, DPQn, DQQn
    #
    def _Jac_forward_both(self, P, Q, no_steps):
        """
        Diffeo: Jacobian d(q,p)^(no_steps)/d(p0,q0) #
        """
        # initial p derivs
        DPP =np.eye(self.l1)
        DPP =DPP.reshape(self.s1)
        DQP =np.zeros(self.s1)
        # initial q derivs
        DPQ =np.zeros(self.s1)
        DQQ =np.eye(self.l1)
        DQQ =DQQ.reshape(self.s1)
        # time stepping
        for i in range(no_steps):
            DPP,DQP,DPQ,DQQ = self._one_step_Jac_both(P, Q, DPP, DQP, DPQ, DQQ)
            P, Q = self._one_step(P, Q)
        return DPP, DQP, DPQ, DQQ
    #
    def _one_step_Jac(self, P, Q, DPP, DQP):
        """
        Diffeo: Apply one step for Jacobian: d/dp only
        """
        HPP=self.Dpp(P, Q)
        HPQ=self.Dpq(P, Q)
        HQQ=self.Dqq(P, Q)
        DPPn=DPP-(self._contract_t(HPQ,DPP) + self._contract(HQQ,DQP))*self.dt
        DQPn=DQP+(self._contract(HPP,DPP) + self._contract(HPQ,DQP))*self.dt
        return DPPn, DQPn
    #
    def _Jac_forward_p(self, P, no_steps):
        """
        Diffeo: Jacobian d(q,p)^(no_steps)/d(p0)
        starting from Q=Qr
        """
        Q=np.copy(self.landmarks[0,:,:])
        P=P.reshape(self.s0)
        #
        DPP=np.eye(self.l1)
        DPP=DPP.reshape((self.s1))
        DQP=np.zeros((self.s1))
        # time stepping
        for i in range(no_steps):
            DPP, DQP =self._one_step_Jac(P, Q, DPP, DQP)
            P, Q   =self._one_step(P, Q)
        return DPP, DQP
    #
    def _Jac_forward_all(self, P, Q, no_steps):
        """
        Diffeo: Jacobian d(q,p)^(no_steps)/d(pin,qin)
        """
        # initial p derivs
        DPP =np.eye(self.l1)
        DPP =DPP.reshape(self.s1)
        DQP =np.zeros(self.s1)
        # initial q derivs
        DPQ =np.zeros(self.s1)
        DQQ =np.eye(self.l1)
        DQQ =DQQ.reshape(self.s1)
        # time stepping
        for i in range(no_steps):
            DPP, DQP, DPQ, DQQ = self._one_step_Jac_both(P, Q,
                                                         DPP, DQP, DPQ, DQQ)
            P, Q = self._one_step(P, Q)
        return DPP, DQP, DPQ, DQQ, P, Q


    # Overload print operation
    def __str__(self):
        """
        Diffeo:
        """
        s  = "--\nDiffeo: "+ Hamiltonian.__str__(self)
        return s
    #
    def _contract(self,A,B):
        """
        Diffeo:
        """
        return np.tensordot(A,B,axes=([2,3],[0,1]))
    #
    def _contract_t(self,A,B):
        """
        Diffeo:
        """
        return np.tensordot(A,B,axes=([0,1],[0,1]))
    #
#
#
############################
#############################
class Shoot(Diffeo):
    """
    Shoot: Shooting method (Hamiltonian dynamics, exact landmarks)
    """
    def __init__(self, G, report_flag=False):
        Diffeo.__init__(self, G, report_flag)
        self.xtol=1e-6 # tolerance for optimiser
        if self.report_flag:
            self._Jacobian_checker()
    #
    def solve(self):
        """
        Shoot: compute initial momentum to satisfy BVP
        by solving nonlinear system
        """
        print("Shoot: Solving with ",self.no_steps, "steps...")
        sys.stdout.flush()
        self._init_path()
        Pin=np.zeros(self.s0)
        #
        opt={}
        opt['xtol']=self.xtol
        opt['factor']=10
        #
        start=timer()
        Pout=spo.root(self._objective,  np.ravel(Pin),
                      jac=self._Jacobian,  options=opt)
        #
        if Pout['success']==False:
            print("Heavy artillery brought in...")
            sys.stdout.flush()
            P=Pout['x'].reshape(self.s0)
            opt['maxiter']=int(1e4)
            Pout=spo.root(self._objective,  np.ravel(P),
                          jac=self._Jacobian, method='lm',
                          options=opt)
        end=timer()
        print("Run time %3.1f secs" % (end-start))
        #
        print(Pout['message'], "Success=", Pout['success'])
        assert Pout['success']
        P0=Pout['x'].reshape(self.s0)
        self.set_path(P0, self.landmarks[0,:,:])
        return P0
    #
    def _objective(self, Pin):
        """
        Shoot: for nonlinear solve
        """
        P=np.copy(Pin)
        P=P.reshape(self.s0)
        Q=np.copy(self.landmarks[0,:,:]) # get reference landmarks
        P,Q=self.forward(P,Q,self.no_steps) # shoot
        out=Q-self.landmarks[1,:,:] # compare to target
        return np.ravel(out)
    #
    def _Jacobian(self, P):
        """
        Shoot:
        """
        Q=np.copy(self.landmarks[0,:,:])
        P=P.reshape(self.s0)
        DPP,DQP=self._Jac_forward_p(P,self.no_steps)
        return DQP.reshape((self.l1,self.l1))
    #
    def _Jacobian_checker(self):
        """
        Shoot:
        """
        j=np.random.random_integers(0,self.d-1)
        k=np.random.random_integers(0,self.N-1)
        delta=1e-8
        Pr=np.random.normal(0,1,self.s0)
        #
        F=self.objective(np.copy(Pr))
        e=np.zeros(Pr.shape)
        e[j,k]=delta;
        Fd=self.objective(np.copy(Pr+e))
        J1=(Fd-F)/delta
        J1=J1.reshape(Pr.shape)
        J=self.Jacobian(np.copy(Pr))
        J=J.reshape((Pr.shape[0],Pr.shape[1],Pr.shape[0],Pr.shape[1]))
        J1a=J[:,:,j,k]
        if(np.linalg.norm(J1-J1a)>self.TOL*np.linalg.norm(J1a)):
            print("Jacobian", J1a, "\n Column approx", J1)
            print("Jacobian checker error: ",np.linalg.norm(J1-J1a))
        assert(np.linalg.norm(J1-J1a)<self.TOL*np.linalg.norm(J1a))
        print("Passed Jacobian checks in Shoot")
#
#
############################################################
class MultiShoot(Diffeo):
    """
    Multiple-shooting method
    """
    def __init__(self, G, report_flag=False):
        """
        MultiShoot: Shooting method (Hamiltonian dynamics, exact landmarks)
        with multiple shooting
        """
        Diffeo.__init__(self, G, report_flag)
        self.xtol=1e-7 # tolerance for optimiser
        self.scale=1e5 # scaling for p variables
        self.set_no_steps([5,5])  # default
        print("MultiShoot: Scale for p=", self.scale)

    #
    def solve(self):
        """
        MultiShoot:
        """
        print("MultiShoot: solve with steps", self.m_no_steps,
              " and total of ",  self.no_steps, " steps...")
        sys.stdout.flush()
        self._init_path()
        noVars=2*self.m_no_steps.size-1
        # set-up the initial guess
        u0=np.zeros((noVars, self.d, self.N))
        diag=np.ones((noVars, self.d, self.N))
        diag[::2,:,:]=self.scale
        if self.initialguess==1:
            # case 1
            print("Initial guess: flow")
            sys.stdout.flush()
            Q=np.copy(self.landmarks[0,:,:])
            P=np.zeros((self.d,self.N))
            for i in range(1,self.m_no_steps.size-1):
                for k in range(self.no_steps):
                    P, Q=self._one_step(P, Q)
                u0[2*i-1,:,:]=Q
                u0[2*i,:,:]=P
                i=self.m_no_steps.size-1
                P,Q=self.forward(P,Q,self.m_no_steps[i])
                u0[2*i-1,:,:]=Q
        elif self.initialguess==0:
            #case 0
            print("Initial guess: linear in q")
            sys.stdout.flush()
            j=0
            delta=self.landmarks[1,:,:]- self.landmarks[0,:,:]
            for i in range(0,self.m_no_steps.size-1):
                j=j+self.m_no_steps[i]
                u0[2*i+1,:,:]=self.landmarks[0,:,:] + delta * (j/self.no_steps)
        # allocate memory for return
        uout=np.zeros(self.d* self.N* noVars)
        # optimize
        opt={}
        opt['xtol']=self.xtol
        opt['factor']=10
        opt['diag']=np.ravel(diag)
        start=timer()
        Pout=spo.root(self._objective, u0,
                      jac=self._Jacobian,
                      options=opt)
        if Pout['success']==False:
            print("Heavy artillery brought in...")
            sys.stdout.flush()
            opt['maxiter']=int(1e4)
            Pout=spo.root(self._objective, Pout['x'],
                          jac=self._Jacobian,
                          method='lm',    options=opt)
        end=timer()
        print("Run time %3.1f secs" % (end-start))
        print(Pout['message'], "Success=", Pout['success'])
        assert Pout['success']
        X=np.array(Pout['x']).reshape((noVars, self.d, self.N))
        P0=X[0,:,:]
        #
        self.set_path(P0,self.landmarks[0,:,:])
        return P0
    #
    def _objective(self, uin):
        """
        MultiShoot:
        uin is a vector, that is immediately reshaped so
        uin[i,j,k]=i dimension, j particle number,
        k=intermediate shoot params in
        order (p;q,p;q,...;p,q;).
        each semi-colon is a shoot
        """
        noVar=2*self.m_no_steps.size-1
        # reshape current state
        uin=uin.reshape((noVar, self.d, self.N))
        # allocate memory for return
        out=np.zeros_like(uin)
        # initial shoot
        i=0;  j=0
        Q=np.copy(self.landmarks[0,:,:])
        P=np.copy(uin[j,:,:])
        P,Q=self.forward(P,Q,self.m_no_steps[i])
        #
        out[j,:,:]  =Q-uin[j+1,:,:]
        out[j+1,:,:]=(P-uin[j+2,:,:])
        # intermediate shoots
        if (self.m_no_steps.size>2):
            for i in range(1,self.m_no_steps.size-1):
                j=2*i
                Q=np.copy(uin[j-1,:,:])
                P=np.copy(uin[j  ,:,:])
                P,Q=self.forward(P,Q,self.m_no_steps[i])
                out[j  ,:,:] =Q-uin[j+1,:,:]
                out[j+1,:,:] =(P-uin[j+2,:,:])
        # final shoot
        i=self.m_no_steps.size-1
        j=2*i
        Q=np.copy(uin[j-1,:,:])
        P=np.copy(uin[j  ,:,:])
        P,Q=self.forward(P,Q,self.m_no_steps[i])
        #
        out[j,:,:]=(Q-self.landmarks[1,:,:])
        #
        out=out.ravel()
        if self.report_flag:
            print(np.linalg.norm(out))
        return out
    #
    def _Jacobian(self, uin):
        """
        MultiShoot:
        """
        noVars=2*self.m_no_steps.size-1
        # reshape current state
        uin=uin.reshape((noVars, self.d, self.N))
        # define usefully sized identity matrix
        id=np.eye(self.l1)
        id=id.reshape(self.s1)
        # allocate memory for return
        bigJ=np.zeros((noVars, self.d,self.N, noVars, self.d,self.N))
        # initial shoot
        i=0;        j=0
        P=np.copy(uin[j,:,:])
        DPP, DQP=self._Jac_forward_p(P,self.m_no_steps[i])
        # out[:,:,j]  =Q-uin[:,:,j+1]
        bigJ[j,:,:,j,:,:]=DQP # dq/dp
        bigJ[j,:,:,j+1,:,:]=-id #
        # out[:,:,j+1]=(P-uin[:,:,j+2])
        bigJ[j+1,:,:,j,:,:]=DPP # dp/dp
        bigJ[j+1,:,:,j+2,:,:]=-id #
        # intermediate shoots
        for i in range(1,self.m_no_steps.size-1):
            j=2*i
            Q=np.copy(uin[j-1,:,:])
            P=np.copy(uin[j,:,:])
            DPP,DQP,DPQ,DQQ=self._Jac_forward_both(P,Q,self.m_no_steps[i])
            # out[:,:,j]  =Q-uin[:,:,j+1]
            bigJ[j,:,:,j-1,:,:]=DQQ # dq/dq
            bigJ[j,:,:,j,:,:]=DQP   # dq/dp
            bigJ[j,:,:,j+1,:,:]=-id
            #  out[:,:,j+1]=(P-uin[:,:,j+2])
            bigJ[j+1,:,:,j-1,:,:]=DPQ # dp/dq
            bigJ[j+1,:,:,j,:,:]  =DPP   # dp/dp
            bigJ[j+1,:,:,j+2,:,:]=-id
        # final shoot
        i=self.m_no_steps.size-1
        j=2*i
        Q=np.copy(uin[j-1,:,:])
        P=np.copy(uin[j,:,:])
        DPP,DQP,DPQ,DQQ=self._Jac_forward_both(P,Q,self.m_no_steps[i])
        # out[:,:,j]=Q-self.Qt
        bigJ[j,:,:,j-1,:,:]=DQQ # dq/dq
        bigJ[j,:,:,j,:,:]  =DQP # dq/dp
        #
        bigJ=bigJ.reshape((self.l1*noVars,self.l1*noVars))
        return bigJ
    #
    def _Jacobian_checker(self):
        """
        MultiShoot:
        """
        noVar=2*self.m_no_steps.size-1
        ell=np.random.random_integers(0,noVar-1)
        j=np.random.random_integers(0,self.d-1)
        k=np.random.random_integers(0,self.N-1)
        #
        delta=1e-8
        Pr=np.random.normal(0,1,(noVar,self.d,self.N))
        F=self._objective(np.copy(Pr))
        e=np.zeros(Pr.shape)
        e[ell,j,k]=delta;
        Fd=self._objective(np.copy(Pr+e))
        J1=(Fd-F)/delta
        J1=J1.reshape(Pr.shape)
        J=self._Jacobian(np.copy(Pr))
        J=J.reshape(np.concatenate((Pr.shape,Pr.shape)))
        J1a=J[:,:,:,ell,j,k]
        if(np.linalg.norm(J1-J1a)>self.TOL*np.linalg.norm(J1a)):
            print("Jacobian", J1a, "\n Column approx", J1)
            print("mJacobian checker error: ",np.linalg.norm(J1-J1a))
        assert(np.linalg.norm(J1-J1a)<self.TOL*np.linalg.norm(J1a))
        print("Passed multiple-shoot Jacobian checks")
