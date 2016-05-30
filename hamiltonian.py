from __future__ import (absolute_import, division, #
                        print_function, unicode_literals)
exec(open("ground.py").read())
"""
hamiltonian.py
--------------
Define Green's function and Hamiltonian for image registration, providing gradients and Hessians.

Classes:
GaussGreen: exp(-r^2/ell^2) Green's function (r=inter-particle distance, ell parameter)
Hamiltonian: sum_ij p_i p_j G(q_i, q_j) #

"""
import utility
#
class GaussGreen:
    """
    Class:  GaussGreen(ell,report_flag)
    Gaussian Green function e(-norm(x-y)^2/l^2) in dimension 2
    with length scale ell (default ell=1)
    Provides:
    G(x,y) =evaluates Green's function
    Dx(x,y)=gradient
    Dxx(x,y)=matrix of second x derivatives
    Dxy(x,y)=matrix of (x,y) derivivates
    """
    def __init__(self,ell,report_flag=False):
        """
        GaussGreen: ell= length scale;
        """
        self.report_flag=report_flag
        self.d = 2 # dimension fixed
        self.set_length(ell)
        self.TOL=1e-6  # for derivative checker
        self._check_deriv_x()
        self._check_deriv_x_G2()
        self._check_deriv_x_expG2()
        self._check_deriv_xx()     #
        self._check_deriv_xx_G2_r()
    #
    def __str__(self):
        """
        GaussGreen:
        """
        s = "GaussGreen with d=" + str(self.d)+" "
        s += "and length scale " + str(self.ell)
        return s
    #
    def set_length(self, ell):
        """
        GaussGreen: set length scale
        """
        self.ell=ell
        self.ell_sq=ell*ell
        if self.report_flag:
            print("length scale=",ell)
    #
    def G(self,vx,vy):
        """
        GaussGreen: evaluate the Green's function for vx,vy 2-vectors
        of positions.
        If vx,vy contains N particles (2xN array)
        and vx contains 1 particle (2x1) array,
        then return array of G(vx,vy[i]) for i in range(N)
        """
        d=vx-vy
        r_sq=np.sum(d**2,axis=0)/self.ell_sq
        return np.exp(-r_sq)
    #
    def G2(self,vx,vy):
        """
        GaussGreen: evaluate the Green's function.
        If vx,vy are 2xN and 2xM arrays of particle positions
        then returns a 2d array of G(vx[i],vy[j])
        """
        vxn=vx.size//2; vyn=vy.size//2
        d=vx.reshape((2,vxn,1))-vy.reshape((2,1,vyn))
        r_sq=np.sum(d**2,axis=0)/self.ell_sq
        return np.exp(-r_sq)
    #
    def DG2(self,Q):
        """
        GaussGreen: Compute derivate of Green's fn matrix G(q_i,q_j)
        Returns tensor [i,j,k,l]
        [i,j]=matrix;
        [k,l]=derivative direction with k=space dimension, l particle
        """
        N=Q.shape[1]
        A=self.Dx(Q,Q)
        out=np.zeros((N,N,self.d,N))
        for i in range(N):
            out[i,:,0,i]=A[0,i,:]
            out[:,i,0,i]=A[0,i,:]
            out[i,:,1,i]=A[1,i,:]
            out[:,i,1,i]=A[1,i,:]
        return out
    #
    def expG2(self,q,alpha):
        """
        GaussGreen: matrix exponential
        """
        return scipy.linalg.expm(-alpha*self.G2(q,q))
    #
    def dexpG2(self,q,alpha):
        """
        GaussGreen: derivative of matrix exponential of G2(q)
        [i,j,k,l]
        [i,j]=matrix; [k,l]=direction
        """
        N=q.shape[1]
        A=-alpha*self.G2(q,q)
        B=-alpha*self.DG2(q)
        out=np.zeros((N,N,self.d,N))
        for i in range(self.d):
            for j in range(N):
                out[:,:,i,j]=scipy.linalg.expm_frechet(A,
                                                       B[:,:,i,j],
                                                       compute_expm=False)
        return out, scipy.linalg.expm(A)
    #
    def G2_r(self, Q, r):
        """
        GaussGreen: Evaluate G(Q)*r
        G NxN, r 2xN, output 2xN
        """
        return np.tensordot(r,self.G2(Q,Q), axes=((1), (1)) )
    #
    def D_G2_r(self, Q, r):
        """
        GaussGreen: Evaluate Dq G(q)*r
        output 2xNx2N [i,j,k,l]
        (i,j) G2_r variables; (k,l) direction variables.
        """
        d=Q.shape[0]
        N=Q.shape[1]
        A=self.Dx(Q,Q)
        C=np.tensordot(r,A,axes=((1),(2)))
        B=A.reshape((1,d,N,N))*r.reshape((d,1,N,1))
        B=np.transpose(B,(0,3,1,2))
        for i in range(N):
            B[:,i,:,i]+=C[:,:,i] # use einsum
        return B
    #
    def D2_G2_r(self, Q, r):
        """
        GaussGreen: Evaluate D2q G(q)r
        return 2x2xNx2xN=(i,j,k,l,m)
        (l,m) (j,k) derivative directions
        (i,j)=G(q) r direction
        Yes, it's repeated.
        """
        d=Q.shape[0]
        N=Q.shape[1]
        A=self.Dxx(Q,Q) # 2xNx2xN
        C=np.einsum('ij,klmj->iklm',r,A)
        B=A.reshape((1,d,N,d,N))*r.reshape((d,1,N,1,1))
        #B=np.transpose(B,(0,1,4,3,2))
        B=np.transpose(B,(0,3,2,1,4))
        for i in range(N):
            B[:,:,i,:,i]-=C[:,:,i,:] # use einsum
        return -B
    #
    def _check_deriv_x_G2(self):
        """
        GaussGreen: Check the first-order derivatives by finite differences
        """
        N=3
        if self.report_flag:
            print("Check first-order derivativs for G2_r")
        vx=np.random.normal(0,1,(self.d,N)) #
        vy=np.random.normal(0,1,(self.d,N))
        #
        ee=np.zeros_like(vx)
        delta=1e-6
        i=np.random.random_integers(0,self.d-1)
        j=np.random.random_integers(0,N-1)
        ee[i,j]=delta
        #
        dG_fd=(self.G2(vx+ee,vx+ee)-self.G2(vx-ee,vx-ee))/(2.0*delta)
        dG_an=self.DG2(vx) # analytical derivative
        #print("\n\ni=",i,", j=",j,"\nfd",dG_fd,"\nan",dG_an[:,:,i,j],"\n\n")
        error= dG_fd-dG_an[:,:,i,j]
        #
        assert np.linalg.norm(error)<self.TOL, print("First derivative check failed with error", np.linalg.norm(error))
        if self.report_flag:
            print( "   Passed first-order derivative check in GaussGreen") #

    #
    def _check_deriv_x_expG2(self):
        """
        GaussGreen: Check the first-order derivatives by finite differences
        """
        N=3
        alpha=1
        if self.report_flag:
            print("Check first-order derivativs for expG2")
        vx=np.random.normal(0,1,(self.d,N)) #
        vy=np.random.normal(0,1,(self.d,N))
        #
        ee=np.zeros_like(vx)
        delta=1e-6
        i=np.random.random_integers(0,self.d-1)
        j=np.random.random_integers(0,N-1)
        ee[i,j]=delta
        #
        dG_fd=(self.expG2(vx+ee,alpha)-self.expG2(vx-ee,alpha))/(2.0*delta)
        dG_an,tmp=self.dexpG2(vx,alpha) # analytical derivative
        #print("\n\ni=",i,", j=",j,"\nfd",dG_fd,"\nan",dG_an[:,:,i,j],"\n\n")
        error= dG_fd-dG_an[:,:,i,j]
        #
        assert np.linalg.norm(error)<self.TOL, print("First derivative check failed with error", np.linalg.norm(error))
        if self.report_flag:
            print( "  Passed first-order derivative check in Gauss Green (expG2) ")


    #
    def _check_deriv_x_G2_r(self):
        """
        GaussGreen: Check the first-order derivatives by finite differences
        """
        N=3
        if self.report_flag:
            print("Check first-order derivativs for G2_r")
        vx=np.random.normal(0,1,(self.d,N)) #
        vy=np.random.normal(0,1,(self.d,N))
        #
        ee=np.zeros_like(vx)
        delta=1e-6
        i=np.random.random_integers(0,self.d-1)
        j=np.random.random_integers(0,N-1)

        ee[i,j]=delta
        #
        dG_fd=(self.G2_r(vx+ee,vy)-self.G2_r(vx-ee,vy))/(2.0*delta)
        dG_an=self.D_G2_r(vx,vy) # analytical derivative
        #print("fd",dG_fd,"\nan off",dG_an[:,:,i,j])
        error= dG_fd-dG_an[:,:,i,j]
        #
        assert np.linalg.norm(error)<self.TOL, print("First derivative check failed with error", np.linalg.norm(error))
        if self.report_flag:
            print( "  Passed first-order derivative check in GaussGreen (G2_r)")

    #
    def _check_deriv_xx_G2_r(self):
        """
        GaussGreen: Check the second-order derivatives
        by finite differences for G2_r
        """
        if self.report_flag:
            print("Check second-order derivativs for G2_r")
        N=3
        #
        vx=np.random.normal(0,1,(self.d,N))
        vy=np.random.normal(0,1,(self.d,N))
        #
        ee=np.zeros((self.d,N))
        delta=1e-8
        i=np.random.random_integers(0,self.d-1)
        j=np.random.random_integers(0,N-1)

        ee[i,j]=delta
        #
        d2G_fd=(self.D_G2_r(vx+ee,vy)-self.D_G2_r(vx-ee,vy))/(2.0*delta)
        d2G_an=self.D2_G2_r(vx,vy)
        #
        #print(i,j,"fd",d2G_fd[:,j,:,:],"\nan off",d2G_an[:,i,j,:,:])
        #error= d2G_fd[:,j,:,:]-d2G_an[:,i,j,:,:]
        error= d2G_fd[:,j,:,:]-d2G_an[:,:,:,i,j]
        #print("error", error)#
        assert np.linalg.norm(error)<self.TOL, print("Second-derivative check failed with error ", error)
        if self.report_flag:
            print("   Passed second-order derivative check in GaussGreen (G2_r)")
    #
    def _check_deriv_x(self):
        """
        GaussGreen: Check the first-order derivatives by finite differences
        for G
        """
        vx=np.random.normal(0,1,self.d)
        vy=np.random.normal(0,1,self.d)
        #
        ee=np.zeros((self.d))
        delta=1e-8
        i=np.random.random_integers(0,self.d-1)
        ee[i]=delta
        #
        dG_fd=(self.G(vx+ee,vy)-self.G(vx-ee,vy))/(2.0*delta)
        dG_an=self.Dx(vx,vy) # analytical derivative
        error= dG_fd-dG_an[i]
        #
        assert np.linalg.norm(error)<self.TOL, print("First derivative check failed with error", np.linalg.norm(error))
        if self.report_flag:
            print( "Passed first-order derivative check in GaussGreen (G)")
    #
    def _check_deriv_xx(self):
        """
        GaussGreen: Check the second-order derivatives by finite differences
        """
        N=3
        #
        vx=np.random.rand(self.d,1)
        vy=np.random.rand(self.d,N)
        #
        ee=np.zeros((self.d,1))
        delta=1e-8
        i=np.random.random_integers(0,self.d-1)
        ee[i,0]=delta
        #
        d2G_fd=(self.Dx(vx+ee,vy)-self.Dx(vx-ee,vy))/(2.0*delta)
        d2G_an=self.Dxx(vx,vy)
        error= d2G_fd-d2G_an[:,:,i,:]
        #
        assert np.linalg.norm(error)<self.TOL, print("Second-derivative check failed with error ", error)
        if self.report_flag:
            print("Passed second-order derivative check in GaussGreen (new)")
    #
    def Dx(self, vx, vy):
        """
        GaussGreen: evaluate the x gradient of the Green's function
        if vx 2xN and vy 2xM of positions, return 2xNxM tensor
        Gives [i,j,k] D_{q_j^i} G(q_j,q_k) i=spatial dimension
        """
        vxn=vx.size//2; vyn=vy.size//2
        d=vx.reshape((2,vxn,1))-vy.reshape((2,1,vyn))
        return d*(-2/self.ell_sq*self.G2(vx,vy))
    #
    def Dxx(self, vx, vy):
        """
        GaussGreen: evaluate the 2nd derivative of the Green's function
        return 2xNx2xM array
        """
        two_over_ell_sq=2/self.ell_sq
        eye=np.eye(2).reshape((2,1,2,1))
        vxn=vx.size//2;        vyn=vy.size//2
        sdelta=two_over_ell_sq*(vx.reshape((2,vxn,1))-vy.reshape((2,1,vyn)))
        tmp=sdelta.reshape((2,1,vxn,vyn))*sdelta.reshape((1,2,vxn,vyn))
        tmp2=self.G2(vx,vy).reshape((1,vxn,1,vyn))
        return (tmp.transpose((0,2,1,3))  -two_over_ell_sq*eye)*tmp2
    #
    # def Dxy(self, vx, vy):
    #     """
    #     evaluate the x y derivative of the Green's fn
    #     """
    #     return -self.Dxx(vx,vy)
#
#
######################################################
"""
Class:  Hamiltonian(G,report_flag)
G=Green's fn
Provides:
 H, Dp,  Dq, Dpp, Dqq, Dpq (to compute Hamiltonian and its derivatives)
 sample_Gibbs_for_p, Gibbs_cov_given_q
"""
class Hamiltonian:
    def __init__(self, G, report_flag=False):
        """
        Hamiltonian: init fn via  G=Green's fn
        """
        self.report_flag=report_flag
        self.G=G   # Green's fn
        self.d=G.d # dimension
        self.precompute=False
        # for checking derivs
        self.N = 3 # number of particles
        self.TOL=1e-6 # tolerance for deriv checks
        self._check_deriv()
        self._check_deriv2()
        self._check_deriv_OU_fun()
    #
    def __str__(self):
        """
        Hamiltonian:  Overload print operation
        """
        s = "Hamiltonian with "+str(self.N)+" particles and Green's function: "+str(self.G)
        return s
    #
    def _check_dim(self, P, Q):
        """
        Hamiltonian: check dimensions
        """
        assert P.shape[0]==self.d
        assert Q.shape[0]==self.d
        assert Q.shape[1]==P.shape[1]
    #
    def _check_deriv(self):
        """
        Hamiltonian: Check first-order derivatives using finite difference
        for H
        """
        P=np.random.normal(0,1,(self.d,self.N))
        Q=np.random.normal(0,1,P.shape)
        #
        ee=np.zeros_like(P)
        delta=1e-7
        i=np.random.random_integers(0,self.d-1)
        j=np.random.random_integers(0,self.N-1)
        ee[i,j]=delta
        #
        Hq_fd=(self.H(P,Q+ee)-self.H(P,Q-ee))/(2.0*delta)
        Hp_fd=(self.H(P+ee,Q)-self.H(P-ee,Q))/(2.0*delta)
        #
        Hp_an=self.Dp(P,Q)
        Hq_an=self.Dq(P,Q)
        #
        assert abs(Hp_an[i,j]-Hp_fd)<self.TOL, print("error in p deriv",Hp_en[i,j],Hp_fd)
        assert abs(Hq_an[i,j]-Hq_fd)<self.TOL, print("error in q deriv",Hq_en[i,j],Hq_fd)
        if self.report_flag:
            print("Passed first-order derivative checks in Hamiltonian (H)")
    #
    def _check_deriv2(self):
        """
        Hamiltonian: Check second-order derivatives using finite difference
        """
        P=np.random.normal(0,1,(self.d,self.N))
        Q=np.random.normal(0,1, P.shape)
        #
        ee=np.zeros_like(P)
        i=np.random.random_integers(0,self.d-1)
        j=np.random.random_integers(0,self.N-1)
        delta=1e-7
        ee[i,j]=delta
        # finite differences
        Hpq_fd=(self.Dp(P,Q+ee)-self.Dp(P,Q-ee))/(2.0*delta)
        Hpp_fd=(self.Dp(P+ee,Q)-self.Dp(P-ee,Q))/(2.0*delta)
        Hqq_fd=(self.Dq(P,Q+ee)-self.Dq(P,Q-ee))/(2.0*delta)
        # analytical derivatives
        Hpp_an=self.Dpp(P,Q)
        Hqq_an=self.Dqq(P,Q)
        Hpq_an=self.Dpq(P,Q)
        #
        assert np.linalg.norm(Hpp_an[:,:,i,j]-Hpp_fd)<self.TOL, print("pp deriv error")
        assert np.linalg.norm(Hpq_an[:,:,i,j]-Hpq_fd) <self.TOL, print("pq deriv error")
        assert np.linalg.norm(Hqq_an[:,:,i,j]-Hqq_fd)<self.TOL, print("qq deriv error")
        #
        if self.report_flag:
            print("Passed second-derivative checks in Hamiltonian (H)")
    #
    def H(self, P, Q):
        """
        Hamiltonian: evaluate H at (P,Q)
        """
        PPP=np.tensordot(P,P,axes=((0),(0)))
        GQQ=self.G.G2(Q,Q)
        return 0.5*np.sum(PPP*GQQ)
    #
    def Dp(self, P, Q):
        """
        Hamiltonian: evalute P derivative
        """
        GQQ=self.G.G2(Q,Q)
        HP= np.tensordot(P,GQQ,axes=((1),(0)))
        return HP
    #
    def Dq(self, P, Q):
        """
        Hamiltonian: Evaluate Q derivative
        """
        PPP=np.tensordot(P,P,axes=((0),(0)))
        Gx=self.G.Dx(Q,Q)
        HQ=np.sum(PPP*Gx,axis=(-1))
        return HQ
    #
    def Dpp(self, P, Q):
        """
        Hamiltonian: Evaluate PP derivative
        """
        GQQ=self.G.G2(Q,Q)
        N=P.size//2
        HPP=np.kron(GQQ.reshape((1,N,1,N)),
                    np.eye(self.d).reshape((2,1,2,1)))
        return HPP
    #
    def Dpq(self, P, Q):
        """
        Hamiltonian: Evaluate PQ derivative
        """
        N=P.size//2
        GXQ=self.G.Dx(Q,Q)
        HPQ=P.reshape((2,1,1,N))*np.transpose(GXQ,axes=(1,0,2))
        for k in range(N):
            HPQ[:,k,:,k]=np.tensordot(P[:,:],GXQ[:,:,k],axes=((1),(-1)))
        return -HPQ
    #
    def Dqq(self, P, Q):
        """
        Hamiltonian: Evaluate QQ derivative
        """
        #
        PPP=np.tensordot(P,P,axes=((0),(0)))# NxN
        GXX=self.G.Dxx(Q,Q)
        #
        N=P.shape[1]
        HQQ=-PPP.reshape((1,N,1,N))*GXX
        for k in range(N):
            HQQ[:,k,:,k]=(np.tensordot(PPP[k,:],GXX[:,k,:,:],axes=((-1),(-1)))
            -PPP[k,k]*GXX[:,k,:,k])
        return HQQ
    #
    def gradH(self,P,Q):
        N=Q.shape[1]
        grad_p=self.Dp(P,Q)
        grad_q=self.Dq(P,Q)
        return np.concatenate((grad_p,
                               grad_q)).reshape((2,self.d,N))
    #

    def Hessian(self,Ph,Qh):
        # Hessian for H (exact part)
        N=Qh.shape[1]
        H_pp=self.Dpp(Ph,Qh)
        H_pq=self.Dpq(Ph,Qh)
        H_qq=self.Dqq(Ph,Qh)
        Hess_H=np.zeros((2,self.d,N,2,self.d,N))
        Hess_H[0,:,:,0,:,:]=H_pp
        Hess_H[1,:,:,1,:,:]=H_qq
        Hess_H[0,:,:,1,:,:]=H_pq
        Hess_H[1,:,:,0,:,:]=np.transpose(H_pq,(2,3,0,1))
        return Hess_H
    #
    def Gmat(self, Q):
        """
        Hamiltonian: Evaluate matrix of G(qi,qj)
        """
        return self.G.G2(Q,Q)
    #
    def _quad_exp_fun(self, r, q, s, alpha):
        """
        Hamiltonian: Evaluate r^T exp(-alpha G(q)) s
        (exp is a matrix exponential)
        G NxN; r 2xN; s 2xN
        return scalar
        """
        A=self.G.expG2(q,alpha)
        B=np.tensordot(A,s,axes=((1),(1)))
        return np.tensordot(r,B,axes=((0,1),(1,0)) )
    #
    def _Dq_quad_exp_fun(self,r,q,s,alpha):
        """
        Hamiltonian: return q derivative of quad_exp_fun: (2,N) tensor
        """
        dA,tmp=self.G.dexpG2(q,alpha)
        B=np.tensordot(r,dA,axes=((1),(0))) # 2xNx2xN
        return np.tensordot(B,s, axes=((0,1),(0,1)) )
    #
    def _Dr_quad_exp_fun(self,r,q,s,alpha):
        """
        Hamiltonian: return r derivative of quad_exp_fun: (2,N) tensor
        """
        A=self.G.expG2(q,alpha)
        B=np.tensordot(s,A,axes=((1),(1)))
        return B
    #
    def pre_comp_expG2(self,q,alpha):
        self.store_expG2=self.G.expG2(q,alpha)
        self.precompute=True
    def pre_comp_dexpG2(self,q,alpha):
        dA, A=self.G.dexpG2(q,alpha)
        self.store_expG2=A
        self.store_dexpG2=dA
        self.precompute=True
    #################################################
    def OU_fun(self,r,q,p,alpha):
        """
        Hamiltonian: evaluate 0.5* tmp^T tmp for
        tmp=p-exp(-G(q) alpha) r
        """
        if self.precompute:
            A=self.store_expG2
        else:
            A=self.G.expG2(q,alpha)
        tmp=p-np.tensordot(r, A, axes=((1),(1)))
        return 0.5*np.tensordot(tmp,tmp,axes=((0,1),(0,1) ))
    #
    # def _dp_quad_exp_fun2(self,r,q,p,alpha):
    #     """
    #     Hamiltonian: evaluate tmp^T tmp for
    #     tmp=p-exp(-G(q) alpha) r
    #     """
    #     return (p-self._Dr_quad_exp_fun(p,q,r,alpha))
    # #
    # def _dr_quad_exp_fun2(self,r,q,p,alpha):
    #     """
    #     Hamiltonian: evaluate tmp^T tmp for
    #     tmp=q-exp(-G(q) alpha) r
    #     """
    #     return (-self._Dr_quad_exp_fun(p,q,r,alpha)
    #               +self._Dr_quad_exp_fun(r,q,r,2*alpha))
    # #
    # def _dq_quad_exp_fun2(self,r,q,p,alpha):
    #     """
    #     Hamiltonian: evaluate tmp^T tmp for
    #     tmp=q-exp(-G(q) alpha) r
    #     """
    #     return 0.5*(-2*self.Dq_quad_exp_fun(p,q,r,alpha)
    #             +self.Dq_quad_exp_fun(r,q,r,2*alpha))
    #
    # def _OU_factor(self,r,q,p,alpha):
    #     """
    #     Hamiltonian: return tmp as defined by
    #     tmp=p-exp(-G(q) alpha) r
    #     """
    #     tmp=p-np.tensordot(r, self.G.expG2(q,alpha), axes=((1),(1)))
    #     return tmp
    #
    def _grad_OU_factor(self,r,q,p,alpha):
        """
        Hamiltonian: return Jacobian of
        tmp=p-exp(-G(q) alpha) r
        as tmp, d/dp, d/dq, d/dr
        """
        N=p.shape[1]
        d_dp=np.eye(self.N)
        if self.precompute:
            dA=self.store_dexpG2
            A =self.store_expG2
        else:
            dA, A=self.G.dexpG2(q,alpha)
        d_dq=-np.tensordot(r,dA,axes=((1),(0))) # 2xNx2xN
        d_dr=-A
        tmp=p-np.tensordot(r, A, axes=((1),(1)))
        return tmp, d_dr, d_dq, d_dp
    #
    def Hess_OU_GN(self,Ph,Qh,tPh,alpha):
        """
        Hamiltonian: return the Gauss-Newton approx to Hessian of
        of the OU term
        """
        OU, OU_p, OU_q, OU_tp=self._grad_OU_factor(Ph,Qh,tPh,alpha)
        grad_ou_factor=np.zeros((self.d,self.N,3,self.d,self.N))

        grad_ou_factor[0,:,0,0,:]=OU_p
        grad_ou_factor[1,:,0,1,:]=OU_p
        grad_ou_factor[:,:,1,:,:]=OU_q
        grad_ou_factor[0,:,2,0,:]=OU_tp
        grad_ou_factor[1,:,2,1,:]=OU_tp
        #
        return  np.tensordot(grad_ou_factor,
                             grad_ou_factor,
                             axes=([0,1],[0,1]))
    #
    def grad_OU(self,r,q,p,alpha):
        """
        Hamiltonian:
        """
        tmp,d_dr,d_dq,d_dp=self._grad_OU_factor(r,q,p,alpha)
        f= np.tensordot(tmp,tmp,axes=((0,1),(0,1) ))
        df_dr= np.tensordot(tmp,d_dr,axes=((1),(0)))
        df_dp= tmp
        df_dq= np.tensordot(tmp,d_dq,axes=((0,1),(0,1)))
        grad_OU=np.concatenate((df_dr,
                                df_dq,
                                df_dp)).reshape((3,self.d,self.N))
        return grad_OU
    #
    def _check_deriv_OU_fun(self):
        """
        Hamiltonian: Check first-order derivatives using finite difference
        """
        alpha=1
        N=3
        R=np.random.normal(0,1,(self.d,N))
        Q=np.random.normal(0,1,R.shape)
        P=np.random.normal(0,1,R.shape)
        #
        ee=np.zeros_like(Q)
        delta=1e-7
        i=np.random.random_integers(0,self.d-1)
        j=np.random.random_integers(0,N-1)
        ee[i,j]=delta
        #
        #D_fd=(self.quad_exp_fun2(Q+ee,R,S,alpha)
        #      -self.quad_exp_fun2(Q-ee,R,S,alpha))/(2.0*delta)
        #D_fd=(self.quad_exp_fun2(Q,R+ee,S,alpha)
        #       -self.quad_exp_fun2(Q,R-ee,S,alpha))/(2.0*delta)
        D_fd=(self.OU_fun(R,Q,P+ee,alpha)
              -self.OU_fun(R,Q,P-ee,alpha))/(2.0*delta)
        #

        #D_an=self._dp_quad_exp_fun2(R,Q,P,alpha) # fails
        #print(D_an.shape)
        #print("deriv exp_fn: \nfd", D_fd, "\nan", D_an[i,j])
        #
    #    assert abs(D_fd-D_an[i,j])<self.TOL, print("error in exp_fn derivativate ",D_fd,D_an[i,j])

        if self.report_flag:
            print("Not checking derivatives for OU_fun")
            #
    def sample_Gibbs_for_p(self, Q, beta):
        """
        Hamiltonian: Samples from Gibbs distribution for P conditioned on Q
        with given inverse temperature beta
        """
        Gmat=self.Gmat(Q)
        N=Q.shape[1]
        # Sample N(0,Cov) using Gmat=L L^T; sample = L^{-1} xi/  sqrt(beta)
        #cholS = scipy.linalg.cholesky(Gmat, lower=True)
        #xi = np.random.randn(N,1)
        #out1=scipy.linalg.solve_triangular(cholS, xi, lower=True)
        #xi = np.random.randn(N,1)
        #out2=scipy.linalg.solve_triangular(cholS, xi, lower=True)
        invGmat=utility.nearPSD_inv(Gmat,1e-6)
        out1=np.random.multivariate_normal(np.zeros(N), invGmat)
        out2=np.random.multivariate_normal(np.zeros(N), invGmat)
        return np.vstack([out1.T,out2.T])/(sqrt(beta))
    #
    def Gibbs_cov_given_q(self, Q, beta):
        """
        Hamiltonian: Return covariance matrix for Gibbs distribuion given Q.
        temperature beta
        """
        Gmat=self.Gmat(Q)
        assert(beta>0)
        Cov=scipy.linalg.inv(2*beta*Gmat)
        return Cov
    #
