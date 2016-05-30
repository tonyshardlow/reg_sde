from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
exec(open("ground.py").read())
from scipy.spatial import procrustes
"""
utility.py
--------------
A few helpful routines for plotting


TS, Jan 2016 (made compatible with vectorized hamiltonian.py)

"""
from matplotlib.lines import Line2D
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.path as mpath
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection
from scipy.interpolate import splrep,splev
#
import matplotlib as mpl
import pylab as plt
#
def smooth(v,periodic=True,bigsize=100):
    """
    Take landmark data and return intermediate points,
    so a smooth interpolating curve can be plotted (via splines)
    """
    if periodic==True:
        vn=np.zeros(v.size+1)
        vn[0:v.size]=np.ravel(v)
    else:
        vn=np.ravel(v)
    #
    s=0.0 #interpolation
    vrange=np.linspace(0, 1, vn.size)
    if periodic==True:
        tck = splrep(vrange, vn, s=s, per=1)
    else:
        tck = splrep(vrange, vn, s=s)
    nrange=np.linspace(0, 1, bigsize)
    return splev(nrange, tck)
#
def plot_reference(u,shadow=0,lw=0.5,ms=4):
    # use . as marker
    X=u[0,:]
    Y=u[1,:]
    alpha=1
    include_marker=True
    if shadow==1:
        cstr='k'
        include_marker=False
    elif shadow==2:
        cstr='0.75'
        alpha=0.5
    elif shadow==3:
        cstr='0.2'
        alpha=0.95
    elif shadow==5:
        cstr='0'
        alpha=0.6
    elif shadow==7:
        cstr='k'
        include_marker=False
    else:
        cstr='k'
    #
    plt.plot(smooth(X),smooth(Y),'c-',alpha=alpha,color=cstr,linewidth=lw)
    #
    if include_marker:
        plt.plot(X,Y,'.',markerfacecolor=cstr,alpha=alpha,markeredgewidth=0.,markersize=ms)
#
def plot_average(u,shadow=0,lw=0.5,ms=5):
    X=u[0,:]
    Y=u[1,:]
    if (shadow==1):
        plt.plot(smooth(X),smooth(Y),'y-',linewidth=lw/2)
        plt.plot(X,Y,'y.',alpha=0.3,markeredgewidth=0.,markersize=ms//2)
    elif (shadow==2):
        cstr='g'
        plt.plot(smooth(X),smooth(Y),'-',color=cstr,linewidth=0.5,label='r')
        plt.plot(X,Y,'.',markerfacecolor=cstr   ,markeredgewidth=0.,markersize=ms)
    else:
        cstr="k"#[1., .4, 0.]
        plt.plot(smooth(X),smooth(Y),'-',color=cstr,linewidth=0.5)
        plt.plot(X,Y,'.',markerfacecolor=cstr,markeredgewidth=0.,markersize=ms)
#
def plot_target(u,shadow=0,lw=0.5,ms=4):
    """
    # use * as marker
    """
    X=u[0,:]
    Y=u[1,:]
    alpha=1
    include_marker=True
    if (shadow==1):
        cstr='k'
        include_marker=False
    elif (shadow==2):
        cstr='0.75'
        alpha=0.5
    elif (shadow==3):
        cstr='0.1'
        alpha=0.1
    elif (shadow==5):
        cstr='0'
        alpha=0.6
    elif (shadow==6):
        cstr=[1., .8, 0.]
    elif (shadow==7):
        cstr=[0., .8, 1.]
        include_marker=True
    else:
        cstr=[0., .8, 1.]

    #
    plt.plot(smooth(X),smooth(Y),color=cstr,alpha=alpha,linewidth=lw)
    #
    if include_marker:
        plt.plot(X,Y,'*', markerfacecolor=cstr,
                 markeredgewidth=0., alpha=alpha,markersize=ms)
#
def plot_RT(landmarks,shadow=0,lw=0.5,ms=4):
    plot_target(landmarks[1,:,:],shadow,lw,ms)
    plot_reference(landmarks[0,:,:],shadow,lw,ms)
#
def plot_path(qpath,shadow=0):
    """
    qpath[i,j,k]
    i=spatial dim; j= particle dim; k=time step
    """
    #
    nlines=4
    N=qpath.shape[2]
    no_steps=qpath.shape[0] # no time steps
    step=no_steps//nlines
    for i in range(step,no_steps-step,step):
        plt.plot(smooth(qpath[i,0,:]),smooth(qpath[i,1,:]),
                 '-',color='0.75',linewidth=0.5)
    for i in range(N):
        plt.plot(qpath[:,0,i], qpath[:,1,i],
                 'y-',linewidth=0.5)
    plot_reference(qpath[0,:,:],shadow)
    plot_target(qpath[-1,:,:],shadow)
#
def plot_landmarks(landmarks,shadow=0,lw=0.5,ms=4):
    """
    for multiple landmark sets
    """
    no_lm=landmarks.shape[0]
    for i in range(no_lm):
        plot_reference(landmarks[i,:,:],shadow,lw,ms)
#
def plot_cmp(LM1,LM2):
    """
    draw the difference against landmark index
    of two sets of landmarks
    """
    delta=LM1-LM2
    plt.plot(delta[0,:],'*',markeredgewidth=0.)
    plt.plot(delta[1,:],'*',markeredgewidth=0.)
#
def get_colormap(mymin,mymax):
    """
    """
    cm = mpl.colors.ListedColormap([[0., .4, 1.], [0., .8, 1.],
                                  [1., .8, 0.], [1., .4, 0.]])
    cm.set_over((1., 0., 0.))
    cm.set_under((0., 0., 1.))
    bounds = [-1., -.5, 0., .5, 1.]
    norm = mpl.colors.BoundaryNorm(bounds, cm.N)
    cNorm  = colors.Normalize(vmin=mymin, vmax=mymax)
    scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)
    scalarMap._A = []
    return cm, scalarMap
#
def add_sd_plot(Q,cov,include_color=True ):
    """
    add circles on landmarks to indicate standard deviation
    """
    N=Q.shape[1]
    v=np.zeros(N)
    for i in range(N):
        v[i]=sqrt(cov[i,i])
    mymax=max(v)
    mymin=min(v)
    #
    cm,scalarMap=get_colormap(mymin,mymax)
    rmin=0.01
    for i in range(N):
        tmp1=v[i]
        if include_color:
            colorVal = scalarMap.to_rgba(tmp1)
        else:
            colorVal=[0.2,0.2,0.2]
        # add a circle
        if tmp1>rmin :
            circle = mpatches.Circle(Q[:,i], tmp1,facecolor=colorVal,
                                     alpha=0.8,edgecolor='none')
            plt.axes().add_patch(circle)

    if include_color:
        plt.colorbar(scalarMap)
#
def plot_setup(xinches=2.5,yinches=2,xytype='xy'):
    """
    """
    print("Preparing plot...")
    plt.clf()
    plt.cla()
    plt.axis('auto')
    plt.gcf().set_size_inches(xinches,yinches);
    if xytype=='xy':
        plt.xlabel(r'$x$')
        plt.ylabel(r'$y$')
    if xytype=='x':
        plt.xlabel(r'$x$')
    if xytype=='tx':
        plt.xlabel(r'$t$')
        plt.ylabel(r'$x$')
    if xytype=='ty':
        plt.xlabel(r'$t$')
        plt.ylabel(r'$y$')
#
def get_grid(qr,qt,n,report_flag=False):
    """
    use qr,qt to determine appropriate range.
    Return a grid with n lines (in each direction)
    """
    xc=np.zeros(2)
    yc=np.zeros(2)
    xc[0]=min([np.amin(qr[0,:]),np.amin(qt[0,:])])
    yc[0]=min([np.amin(qr[1,:]),np.amin(qt[1,:])])
    xc[1]=max([np.amax(qr[0,:]),np.amax(qt[0,:])])
    yc[1]=max([np.amax(qr[1,:]),np.amax(qt[1,:])])
    if report_flag:
        print("grid range", xc,yc)
    factor=1.1
    for i in range(2):
        xc[i]=np.sign(xc[i])*factor*abs(xc[i])
        yc[i]=np.sign(yc[i])*factor*abs(yc[i])
    gridx=np.zeros((2*n,n))
    gridy=np.zeros((2*n,n))
    vx=np.linspace(xc[0],xc[1],n)
    vy=np.linspace(yc[0],yc[1],n)
    for i in range(n):
        gridx[i,:]=vx
        gridy[i,:]=np.ones(n)*vy[i]
        gridy[i+n,:]=vy
        gridx[i+n,:]=np.ones(n)*vx[i]
    return gridx, gridy, xc,yc
#
def plot_grid(gridx, gridy):
    """
    """
    n=gridx.shape[0]
    for i in range(n):
        ax=plt.plot(gridx[i,:],gridy[i,:],':',
                    color='0.05',alpha=0.6,linewidth=1.1)
#
def plot_grid2(gridx, gridy):
    """
    """
    n=gridx.shape[0]
    for i in range(n):
        ax=plt.plot(gridx[i,:],gridy[i,:],'*',
                    color='0.05',alpha=0.6,linewidth=1.1)
#
def plot_grid_color2(gx,gy,z):
    """
    """
    mymax=np.max(z)
    mymin=np.min(z)
    cm, scalarMap=get_colormap(mymin,mymax)
    plt.pcolormesh(gx,gy,z,cmap=cm)
    plt.gcf().colorbar(scalarMap)
#
def plot_grid_color(gridx, gridy, color):
    """
    """
    n=gridx.shape[0]
    mymax=max(color)
    mymin=min(color)
    cm, scalaMap=get_colormap(mymin,mymax)
    if gridx.ndim==1:
        gridx=gridx[:,np.newaxis]
        gridy=gridy[:,np.newaxis]
        color=color[:,np.newaxis]
    for i in range(n):
        ax=plt.plot(gridx[i,:],gridy[i,:],':',
                    color='0.05',alpha=0.6,linewidth=1.1)
        for j in range(gridx.shape[1]):
            colorVal = scalarMap.to_rgba(color[i,j])
            ax=plt.plot(gridx[i,j],gridy[i,j],'.',markeredgewidth=0.,
                        color=colorVal)
    plt.gcf().colorbar(scalarMap)
#
def add_noise(landmarks,noise_var):
    """
    Add iid Gaussian noise to each coordinate
    """
    if (noise_var>0):           #
        sd=sqrt(noise_var)
        print("Adding noise to landmarks with variance ", noise_var)
        landmarks = np.copy(landmarks) + np.random.normal(0,sd,landmarks.shape)
    return landmarks
#
def procrust1(landmarks, report_flag=False):
    """
    centre around the origin and apply Procrustes transformation
    relative to first set of landmarks
    """
    l1=landmarks[0,:,:].T
    l1=l1-np.mean(l1,axis=0)
    landmarks[0,:,:]=l1.T
    for i in range(1,landmarks.shape[0]):
        l2=np.copy(landmarks[i,:,:].T)
        l2=l2-np.mean(l2,axis=0)
        R,scale=scipy.linalg.orthogonal_procrustes(l2, l1)
        l2p=np.dot(l2,R)
        landmarks[i,:,:]=l2p.T
    if report_flag:
        print("Landmarks normalised with mean ",np.mean(landmarks,2))
    return landmarks
#
def get_data(fname,step=10):
    """
    load data file (see data/) and recentre and rescale
    """
    print("Loading from ",fname)
    Q = np.loadtxt(fname)
    Q = Q[:,::step]
    Q = Q - (np.mean(Q,axis=1)*np.ones((1,2))).T
    Q = Q/(np.max(Q,axis=1)*np.ones((1,2))).T
    return Q
#
def get_data_circle(r=1,no_points=20):
    """    """
    theta = np.linspace(np.pi/(2.*no_points),
                        2*np.pi,no_points,
                        endpoint=False)*np.ones((1, no_points))
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    return np.matrix(np.concatenate([x,y],axis=0))
#
def get_data_fish(r=1,no_points=20):
    """    """
    theta = np.linspace(np.pi/(2.*no_points),
                        2*np.pi,no_points,
                        endpoint=False)*np.ones((1, no_points))
    x = 2.*np.cos(theta) - 2./np.sqrt(2) * np.sin(theta)**2
    y = 2.*np.cos(theta)*np.sin(theta)
    return np.matrix(np.concatenate([x,y],axis=0))
#
def get_data_sq_ellipse(r=1,k=1,no_points=20):
    theta = np.linspace(np.pi/(2.*no_points),
                        2*np.pi,no_points,
                        endpoint=False)*np.ones((1, no_points))
    r = 1 - 0.5*np.cos(2*theta)
    x = r*np.cos(k*theta)
    y = r*np.sin(k*theta)
    return np.matrix(np.concatenate([x,y],axis=0))
#
def exp1(noise_var,r1=1,r2=2):
    # Concentric circles
    print("exp1 (concentric circles)")
    dict={}
    Qr=get_data_circle(r1)
    Qt=get_data_circle(r2)
    X=np.empty((2,Qr.shape[0],Qr.shape[1]))
    X[0,:,:]=Qr; X[1,:,:]=Qt;
    dict['landmarks'] = procrust1(X)
    # Add noise :)
    dict['landmarks_n']=procrust1(add_noise(dict['landmarks'],noise_var))
    # Set parameters
    dict['ell']=0.5; # Green's
    dict['no_steps']=[5,4]
    dict['lam']=0.01
    dict['beta']=1e3
    dict['data_var']=noise_var
    dict['epsilon_prior']=0.5*1/dict['beta'] # what does this mean??
    dict['fname']="figs/E1_"
    return dict
#
def exp2(noise_var,no_sets=2):
    dict={}
    # circle to squashed ellipse from T&V
    no_points = 20
    # shifted
#    delta=np.array([1,1.5]).reshape(2,1)
#    scale=np.array([2,3]).reshape(2,1)
    # easy
    delta=np.array([0,0]).reshape(2,1)
    scale=np.array([1,1]).reshape(2,1)
    Qr = (get_data_circle(1)+delta)/scale
    #
    k=1; # k=3 more interesting
    Qt = (get_data_sq_ellipse(1,k)+delta)/scale
    #
    X=np.empty((2,Qr.shape[0],Qr.shape[1]))
    X[0,:,:]=Qr; X[1,:,:]=Qt;
    dict['landmarks'] = procrust1(X)
    # Add noise :)
    dict['landmarks_n']=procrust1(add_noise(dict['landmarks'],noise_var))
    # Set parameters
    dict['ell']=0.5
    dict['no_steps']=20#[4,5]
    dict['lam']=0.1 # dissipation
    dict['beta']=1e2 # inverse temperature
    dict['data_var']=noise_var
    dict['epsilon_prior']=0.5*1/dict['beta']# what does this mean??
    dict['fname']="figs/E2_"
    return dict
    #
#
def exp4(noise_var):
    dict={}
    print("exp4 (ellipse,wrap)")
    Qr = get_data('data/wrap2.txt')
    Qt = get_data('data/ellipse2.txt')
    X=np.empty((2,Qr.shape[0],Qr.shape[1]))
    X[0,:,:]=Qr; X[1,:,:]=Qt;
    print(X.shape)
    dict['landmarks'] = procrust1(X)
    # Add noise :)
    dict['landmarks_n']=procrust1(add_noise(dict['landmarks'],noise_var))
    # Set parameters
    dict['ell']=0.50;
    dict['no_steps']=[5,5  ]
    dict['lam']=0.05
    dict['beta']=1e3 # inverse temperature
    dict['data_var']=0.1
    dict['epsilon_prior']=0.5*1/dict['beta']# what does this mean??
    dict['fname']="figs/E4_"
    return dict
#
def exp5(noise_var):
    # Shapes from dataset
    print( "exp5 (cat to dog etc. from datasets)")
    dict={}
    # dict['Qr'] = get_data('data/cat.txt')
    Qr = get_data('data/plane.txt')
    # dict['Qt'] = get_data('data/dog.txt')
    Qt = get_data('data/shark.txt')
    X=np.empty((2,Qr.shape[0],Qr.shape[1]))
    X[0,:,:]=Qr; X[1,:,:]=Qt;
    print(X.shape)
    dict['landmarks'] = X
    # Add noise :)
    dict['landmarks_n']=procrust1(add_noise(dict['landmarks'],noise_var))
    # Set parameters
    dict['ell']=0.2;
    dict['no_steps']=[5,4]
    dict['lam']=0.1
    dict['beta']=0.5 # inverse temperature
    dict['data_var']=0.05
    dict['epsilon_prior']=0.5*1/dict['beta']# what does this mean??
    dict['fname']="figs/E5_"
    return dict
#
def nearPSD(A,epsilon=0):
    # print("Finding nearest spd matrix with parameter ", epsilon)
    eigval, eigvec = np.linalg.eigh(A)
    neweigval=np.maximum(eigval,epsilon)
    out=np.dot(eigvec, np.dot(np.diag(neweigval),eigvec.T))# =A
    return out

def nearPSD_inv(A,epsilon=0):
    # print("Finding nearest inverse spd matrix with parameter", epsilon)
    eigval, eigvec = np.linalg.eigh(A)
    neweigval=np.maximum(eigval,epsilon)
    for i in range(neweigval.size):
        x=neweigval[i]
        if (x>epsilon):
            neweigval[i]=1/x
        else:
            neweigval[i]=0
    out=np.dot(eigvec, np.dot(np.diag(neweigval),eigvec.T))# =A
    return out
