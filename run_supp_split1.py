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
def run(dict,show_source_data=True):
    import os.path
    if 'fname' in dict:
        filename=dict['fname']
    else:
        print("No filename given")
        exit(1)
    print("filename: ",filename," ", dict['ext'])
    G=hamiltonian.GaussGreen(dict['ell'],0)
    no_steps=dict['no_steps']
    if isinstance(no_steps, list):
        SDE=sde.MAP2(G)
    else:
        SDE=sde.MAP1(G)    # use single shooting
    #
    SDE.set_no_steps(no_steps)
    SDE.set_landmarks(dict['landmarks_n'])
    SDE.set_data_var(dict['data_var'])
    SDE.set_lam_beta(dict['lam'],dict['beta'])
    SDE.solve()
     # start fig 1
    plt.figure(1)
    plot_setup()
    plt.axis('equal')
    if show_source_data:
        plot_RT(dict['landmarks_n'],shadow=2)  # light grey data
    SDE.plot_warp(7)
    cov_q,cov_p=SDE.cov()
    add_sd_plot(SDE.Qrn,cov_q)
    plt.savefig(filename+dict['ext']+'.pdf',bbox_inches='tight')
    print("...finished.")
    #
####################################################################

def myset(i):
    include_multipleshoot=False
    include_shoot=True
    betas=np.array([4,4,40,40])
    no_steps=[5,5]
    if i==1:
        def exp (x): return exp1(x)
        scale=1
    if i==2:
        def exp (x): return exp2(x)
        scale=1
        betas=np.array([4,4,100,100])
    if i==4:
        betas=np.array([8,8,100,100])
        def exp (x): return exp4(x)
        scale=1
    if i==5:
        def exp (x): return exp5(x)
        betas=np.array([0.5,0.5,100,100])
        scale=0.25
    #
    noise_vars=np.array([0.005, 0.005, 0.005, 0.005])


    # lam=1 lambda does not affect the split 1 prior
    exts=['split1_a', 'split1_b', 'split1_c', 'split1_d']
    if include_shoot:
        for i in range(noise_vars.shape[0]):
            print("===============================\nLoading data")
            dict=exp( scale*noise_vars[i] )
            dict['ext']=exts[i]
            dict['beta']=betas[i]
            dict['data_var']=scale*noise_vars[i]
            dict['no_steps']=int(np.prod(no_steps))
            run(dict)
    if include_multipleshoot:
        exts=['msl_a', 'msl_b', 'msl_c', 'msl_d']
        for i in range(noise_vars.shape[0]):
            print("===============================\nLoading data")
            dict=exp( scale*noise_vars[i] )
            dict['ext']=exts[i]
            dict['beta']=betas[i]
            dict['no_steps']=no_steps
            run(dict)


if __name__ == "__main__":
    # do this
    plt.ion()
    myset(1)
    myset(2)
    myset(4)
    myset(5)
