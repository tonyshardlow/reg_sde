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
def run(dict):
    import os.path
    if 'fname' in dict:
        filename=dict['fname']
    else:
        print("No filename given")
        exit(1)
    print("filename: ",filename)
    G=hamiltonian.GaussGreen(dict['ell'])
    no_steps=dict['no_steps']
    if isinstance(no_steps, list):
        ODE=diffeo.MultiShoot(G)
    else:
        ODE=diffeo.Shoot(G)  # use single shooting
    #
    ODE.set_no_steps(no_steps)
    ODE.set_landmarks(dict['landmarks_n'])
    ODE.solve()
    # plot warp
    plt.figure(1)
    plot_setup()
    plt.axis('equal')
    plot_RT(dict['landmarks_n'])
    #plot_RT(dict['landmarks'],shadow=2)
    ODE.plot_warp()
    plt.savefig(filename+".pdf",bbox_inches='tight')
    # SDE linearisaion
    SDE = sde.SDELin(ODE)
    SDE.set_lam_beta(dict['lam'],dict['beta'],True)
    SDE.set_lin_path(ODE.Ppath,ODE.Qpath)
    print("Set epsilon_prior equal to data_var")
    epsilon_prior=dict['data_var']
    SDE.set_prior_eps(epsilon_prior)
    SDE.do_all(dict['data_var'])
    # plots in q vs t
    plot_setup(xytype='ty')
    SDE.plot_q()
    plt.savefig(filename+'_sde2.pdf',bbox_inches='tight') # Fig 3 right.
    ####
    plot_setup()
    SDE.sd_plot(True)
    plt.axis('equal')
    plt.savefig(filename+'_sde_sd.pdf',bbox_inches='tight') # Fig 3 left
    ####
    plot_setup()
    plt.axis('equal')
    g0=np.array([[-2,2],[-2,2]]); g1=np.array([[-1,1],[-1,1]]); # range for grid
    nl=20; # number of lines
    noSamples=5;
    gx,gy,xc,yc=get_grid(g0,g1,nl)
    mgx,mgy,sd=SDE.get_grid_stats(gx,gy,noSamples)
    plot_grid_color2(mgx,mgy,sd)
    plot_target(dict['landmarks_n'][1,:,:],shadow=5)
    plt.savefig(filename+'_pf_cond.pdf',bbox_inches='tight') # Fig 4 left
    #####
    plot_setup()
    plt.axis('equal')
    plot_grid_color2(gx,gy,sd)
    plot_reference(dict['landmarks_n'][0,:,:],shadow=5)
    plt.savefig(filename+'_pf_cond2.pdf',bbox_inches='tight') # Fig 4 right
    #
    print("...finished.")
####################################################################
if __name__ == "__main__":
    # do this
    plt.ion()
    noise_var=0.01
    #noise_var=0.05
    dict=exp1(noise_var)
    #dict=exp2(noise_var)
    #noise_var=0.0
    #dict=exp4(noise_var)
    #dict=exp5(noise_var)
    dict['beta']=25
    dict['lam']=0.1
    run(dict)
