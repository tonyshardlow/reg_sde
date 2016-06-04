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
def run(dict,canload=0):
    """
    Qr=dict['landmarks'][0,:,:]
    Apply push-forward map to sample Qr.
    Find average using MAP4.
    Plot phase-plane with confidence balls and original data (from push-forward map).
    """
    import os.path
    if 'fname' in dict:
        filename=dict['fname']
    else:
        print("No filename given")
        exit(1)
    print("\n",filename," ",dict['ext'],"============================================","\n")
    plt.ion()
    G=hamiltonian.GaussGreen(dict['ell'],0)
    # create set of landmarks by push-forward
    SDE = sde.SDE(G)
    SDE.set_lam_beta(dict['lam'],dict['beta'],True)
    Qr=dict['landmarks'][0,:,:] # use first set as reference
    dict['landmarks_n']=SDE.add_sde_noise(Qr, dict['num'])
    # find landmark average
    SDE=sde.MAP4(G)
    SDE.set_data_var(dict['data_var'])
    SDE.set_lam_beta(dict['lam'],dict['beta'],False)
    SDE.set_landmarks(dict['landmarks_n'])
    SDE.set_no_steps(dict['no_steps'])
    SDE.solve()
    cov_q,cov_p=SDE.cov()
    #
    # plot landmarks (noisy source data)
    plt.figure(1)
    plot_setup()
    plt.axis('equal')
    plot_landmarks(dict['landmarks_n'],shadow=3,lw=0.2)
    #plt.savefig(filename+dict['ext']+'_samps.pdf',bbox_inches='tight') # lam
    # plot landmarks with average and confidence ball
    plt.figure(1)
    plot_setup()
    plt.axis('equal')
    plot_average(SDE.Qh) # red star
    Qav=np.average(dict['landmarks_n'],axis=0)
    plot_average(Qav,2) # 2=blue dot
    add_sd_plot(SDE.Qh, cov_q)
    plt.savefig(filename+dict['ext']+'_av.pdf',bbox_inches='tight')

####################################################################

if __name__ == "__main__":
    # do this
    plt.ion()
    #
    noise_var=0.0
    dict=exp4(noise_var)
    dict['beta']=25
    dict['lam']=0.1
    i=2
    if i==1:
        dict['ext']='two'
        dict['num']=2
    if i==2:
        dict['ext']='sixteen'
        dict['num']=16
    dict['data_var']=noise_var+0.05
    run(dict)
