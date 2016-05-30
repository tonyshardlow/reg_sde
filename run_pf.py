from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
exec(open("ground.py").read())
# mine
import hamiltonian
import diffeo
import sde
from utility import *
#
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
    print("filename: ",filename+dict['ext'])
    #
    G=hamiltonian.GaussGreen(dict['ell'],0)
    no_steps=dict['no_steps']
    #
    SDE = sde.SDE(G)
    SDE.set_no_steps(no_steps)
    SDE.set_landmarks(dict['landmarks_n'])
    SDE.set_lam_beta(dict['lam'],dict['beta'],True)
    # plot a push-forward sample (with current shape)
    plot_setup()
    plt.axis('equal')
    plt.axis('off')
    Q0=dict['landmarks'][0,:,:]
    D=SDE.sample_push_forward(Q0)
    D.plot_qpath_01(0)
    D.plot_warped_grid(10)
    plt.savefig(filename+dict['ext']+'.pdf',bbox_inches='tight')
    print("...finished.")
   #
####################################################################
if __name__ == "__main__":
    # do this
    plt.ion()
    noise_var=0.2
    dict=exp1(noise_var)
    #dict=exp2(noise_var)
    #dict=exp4(noise_var)
    #    dict=exp4(noise_var)
    dict['lam']=0.5
    scale=1.0e1;betas=np.array([1., 2., 4.0, 8.])*scale
    exts=['a_pf', 'b_pf', 'c_pf', 'd_pf']
    for i in range(4):
        print("=======")
        dict['beta']=betas[i]
        dict['ext']=exts[i]
        run(dict)
