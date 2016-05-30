from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
exec(open("ground.py").read())
# mine
import hamiltonian
import diffeo
import sde
from utility import *
#
# all data defined in utility (exp2,...)
#
def run(dict,canload=0):
    import os.path
    if 'fname' in dict:
        filename=dict['fname']
    else:
        print("No filename given")
        exit(1)
    print("\n",filename,"======================================","\n")
    G=hamiltonian.GaussGreen(dict['ell'],0)
    # plot target and reference data (original and perturbed)
    plt.figure(1)
    plot_setup()
    plt.axis('equal')
    plot_RT(dict['landmarks_n'])

    # plot_RT(dict['landmarks'],shadow=2)
    #
    no_steps=dict['no_steps']
    if isinstance(no_steps, list):
        ODE=diffeo.MultiShoot(G)
    else:
        ODE=diffeo.Shoot(G)  # use single shooting
    #
    ODE.set_no_steps(no_steps)
    ODE.set_landmarks(dict['landmarks_n'])
    ODE.solve()
    # add on results of shooting
    ODE.plot_warp()
    #    plt.savefig(filename,bbox_inches='tight')
    plt.savefig(filename+'_ic0_'+dict['ext']+'.pdf',bbox_inches='tight') # Fig 5
    # linearisaion
    SDE = sde.SDELin(ODE)
    SDE.set_lam_beta(dict['lam'],dict['beta'],True)
    SDE.set_lin_path(ODE.Ppath,ODE.Qpath)
    #
    print("Set epsilon_prior equal to data_var")
    epsilon_prior=dict['data_var']
    SDE.set_prior_eps(epsilon_prior)
    #
    SDE.do_all(dict['data_var'])
    # show inconsistency linear vs diffeo
    SDE.sample()
    plot_setup()
    plt.axis('equal')
    plt.axis('on')
    # get initial points for conditioned sample
    gx=SDE.Qpath[0,0,:]; gy=SDE.Qpath[0,1,:]
    lref=np.array([gx.T,gy.T])
    wgx,wgy=SDE.diffeo_arrays(gx,gy)
    ltar1=np.array([wgx.T,wgy.T])
    #
    tgx=SDE.Qpath[-1,0,:]; tgy=SDE.Qpath[-1,1,:]
    ltar=np.array([tgx.T,tgy.T])
    #
    plot_reference(lref)
    plot_target(ltar1,shadow=6)
    plot_target(ltar)
    #
    plt.savefig(filename+'_ic_'+dict['ext']+'.pdf',bbox_inches='tight') # Fig 5
####################################################################
if __name__ == "__main__":
    # do this
    plt.ion()

    #noise_vars=np.array([0.005, 0.01, 0.015, 0.02])*0.5 # for exp1
    noise_vars=np.array([0.005, 0.01, 0.015, 0.02]) # for exp2
    exts=['a_pf', 'b_pf', 'c_pf', 'd_pf']
    for i in range(noise_vars.shape[0]):
        print("i=",i,"   ")
        noise_var=noise_vars[i]
        dict=exp2(noise_var)
        dict['ext']=exts[i]
        dict['no_steps']=[5,5]
        #dict['data_var']=0.
        run(dict)
