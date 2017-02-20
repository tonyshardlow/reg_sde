from __future__ import (absolute_import, division,
                        print_function, unicode_literals)
exec(open("ground.py").read())
#
import matplotlib.pyplot as plt
import skimage
from skimage import data
# mine
import hamiltonian
import diffeo
import sde
from utility import *
#

def run(dict,canload=0):
    import os.path
    if 'fname' in dict:
        filename=dict['fname']
    else:
        print("No filename given")
        exit(1)
    print("\n",filename,"============================================","\n")
    plt.ion()
    G=hamiltonian.GaussGreen(dict['ell'],0)
    no_steps=dict['no_steps']
    if isinstance(no_steps, list):
        ODE=diffeo.MultiShoot(G,1)
    else:
        ODE=diffeo.Shoot(G)  # use single shooting
    #
    ODE.set_no_steps(dict['no_steps'])
    ODE.set_landmarks(dict['landmarks_n'])
    ODE.solve()
    # plot warp
    plot_setup()
    plt.axis('equal')
    ODE.plot_warp()
    plt.savefig(filename+'warp.pdf',bbox_inches='tight')
    #
    # load test image
    #image = data.checkerboard()
    #image = data.coffee()
    image = mpl.image.imread('51h2011_1_0.jpg')
    #
    # apply warp to image
    new_image=ODE.warp(image)
    # plotting and save to png
    plot_setup()
    plt.close()
    fig, (ax0, ax1) = plt.subplots(1, 2,
                                   figsize=(8, 3),
                                   sharex=True,
                                   sharey=True,
                                   subplot_kw={'adjustable':'box-forced'}
                                   )
    ax0.imshow(image, cmap=plt.cm.gray, interpolation='none')
    mpl.image.imsave('hand_alt.jpg',image,cmap=plt.cm.gray)
    ax0.axis('off')
    #
    ax1.imshow(new_image, cmap=plt.cm.gray, interpolation='none')
    mpl.image.imsave('hand_new.jpg',new_image,cmap=plt.cm.gray)
    ax1.axis('off')
    plt.show()
    print("finished.")

####################################################################
if __name__ == "__main__":
    # do this
    plt.ion()
    noise_var=0.00
    #dict=exp1(noise_var)
    dict=exp2(noise_var)
    # dict=exp4(noise_var)
    # dict=exp4(noise_var)
    run(dict)
