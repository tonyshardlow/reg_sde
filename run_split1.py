from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
exec(open("ground.py").read())
#

# mine
import hamiltonian
import diffeo
import sde
from utility import *
#
def run(dict,show_source_data=True):
    """
    Uses MAP1, to register two sets of landmarks from dict['landmarks_n']
    Generates plots with dict['filename']+X+dict['ext'] extension
    - X='': phase-space plot, with confidence balls around target
    - X='cmp_': time vs space
    """

    import os.path
    if 'fname' in dict:
        filename=dict['fname']
    else:
        print("No filename given")
        exit(1)
    print("\n",filename,"============================================","\n")
    #
    G=hamiltonian.GaussGreen(dict['ell'],0)
    no_steps=dict['no_steps']
    if isinstance(no_steps, list):
        SDE=sde.MAP2(G)          # Multipleshooting
    else:
        SDE=sde.MAP1(G)          # single shooting
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
    # add discs with one standard deviation
    cov_q,cov_p=SDE.cov()
    add_sd_plot(SDE.Qrn,cov_q)
    plt.savefig(filename+dict['ext']+'.pdf',bbox_inches='tight')
    # plots in q vs t
    plot_setup(xytype='ty')
    plt.ylabel(r'$x, y$')
    plt.xlabel(r'$i$')
    plot_cmp(dict['landmarks_n'][0,:,:],SDE.Qrn )
    plt.savefig(filename+'cmp_'+dict['ext']+'.pdf',bbox_inches='tight')
####################################################################
if __name__ == "__main__":
    # do this
    plt.ion()
    #
    noise_var=0.005
    dict=exp4(noise_var)
    # dict['lam']=  not relevant for split 1 prior
    dict['beta']=25
    dict['data_var']=noise_var
    exts=['a', 'b']
    for i in range(2):
        dict['ext']=exts[i]
        dict['no_steps']=np.prod(dict['no_steps'])
        run(dict)
