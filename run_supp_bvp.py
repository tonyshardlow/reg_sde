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
    print("filename :",filename," ",dict['ext'])
    G=hamiltonian.GaussGreen(dict['ell'],0)
    no_steps=dict['no_steps']
    if isinstance(no_steps, list):
        ODE=diffeo.MultiShoot(G)
    else:
        ODE=diffeo.Shoot(G)  # use single shooting
    ODE.set_no_steps(no_steps)
    ODE.set_landmarks(dict['landmarks_n'])
    ODE.solve()
    # plot
    plt.figure(1)
    plot_setup()
    plt.axis('equal')
    plot_RT(dict['landmarks_n'])
    #plot_RT(dict['landmarks'],shadow=2)
    ODE.plot_warp()
    plt.savefig(filename+'_simple_'+dict['ext']+'.pdf',bbox_inches='tight')
    print("...finished.")
    #
####################################################################

def myset(i):
    include_multipleshoot=True
    include_shoot=True
    no_steps=[5,5]
    if i==1:
        def exp (x): return exp1(x)
        scale=1
    if i==2:
        def exp (x): return exp2(x)
        scale=0.1
    if i==4:
        include_shoot=False
        def exp (x): return exp4(x)
        scale=0.1
    if i==5:
        def exp (x): return exp5(x)
        scale=0.1
    #
    noise_vars=np.array([0.0, 0.01, 0.015, 0.02])
    exts=['xa', 'xb', 'xc', 'xd']
    if include_shoot:
        for i in range(noise_vars.shape[0]):
            print("===============================\nLoading data")
            dict=exp( scale*noise_vars[i] )
            dict['ext']=exts[i]
            dict['no_steps']=int(np.prod(no_steps))
            run(dict)
    if include_multipleshoot:
        exts=['xms_a', 'xms_b', 'xms_c', 'xms_d']
        for i in range(noise_vars.shape[0]):
            print("===============================\nLoading data")
            dict=exp( scale*noise_vars[i] )
            dict['ext']=exts[i]
            dict['no_steps']=no_steps
            run(dict)


if __name__ == "__main__":
    # do this
    plt.ion()
    myset(1)
    myset(4)
    myset(5)
    myset(2)
