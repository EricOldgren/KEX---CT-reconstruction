import numpy as np
import odl

from odl.phantom import ellipsoid_phantom, shepp_logan_ellipsoids
from odl import DiscretizedSpace, DiscretizedSpaceElement
import matplotlib.pyplot as plt

#  #           value  axisx  axisy     x       y  rotation
#     return [[2.00, .6900, .9200, 0.0000, 0.0000, 0],
#             [-.98, .6624, .8740, 0.0000, -.0184, 0],
#             [-.02, .1100, .3100, 0.2200, 0.0000, -rad18],
#             [-.02, .1600, .4100, -.2200, 0.0000, rad18],
#             [0.01, .2100, .2500, 0.0000, 0.3500, 0],
#             [0.01, .0460, .0460, 0.0000, 0.1000, 0],
#             [0.01, .0460, .0460, 0.0000, -.1000, 0],
#             [0.01, .0460, .0230, -.0800, -.6050, 0],
#             [0.01, .0230, .0230, 0.0000, -.6060, 0],
#             [0.01, .0230, .0460, 0.0600, -.6050, 0]]

def random_ellipsoid(min_pt, max_pt, value = 0.5):
    "Generate an ellips in the rectangular region bordered by mint_pt, max_pt"
    (mx, my), (Mx, My) = min_pt, max_pt
    dx, dy = Mx-mx, My-my
    centerx = np.random.uniform(mx+dx*0.05, Mx-dx*0.05)
    centery = np.random.uniform(my+dy*0.05, My-dy*0.05)
    Rx = np.min((Mx-centerx, centerx-mx))
    Ry = np.min((My-centery, centery-my))
    rx, ry = np.random.random()*Rx, np.random.random()*Ry
    #Ellipse
    #        value  axisx  axisy   x    y   rotation(rad)
    return [value, rx,     ry,centerx, centery, np.random.random()*np.pi]

def random_phantom(reco_space: DiscretizedSpace, num_ellipses = 10)->DiscretizedSpaceElement:

    brain_center = np.random.uniform(-0.1, 0.1, 2)
    brain_r = np.random.uniform(0.5, 0.7, 2)
    min_s = np.min(brain_r) / np.sqrt(2)
    skull = [1.0, brain_r[0], brain_r[1], brain_center[0], brain_center[1], 0.0]
    brain = [-0.5, brain_r[0]*0.95, brain_r[1]*0.95, brain_center[0], brain_center[1], 0.0]

    print("Skull: ", skull)
    print("Brain: ", brain)

    ellipses = [skull, brain]
    for i in range(num_ellipses):
        ellipses.append(random_ellipsoid( brain_center-min_s, brain_center+min_s, value = np.random.uniform(0.1, 0.3)))
    
    res = ellipsoid_phantom(reco_space, ellipsoids=ellipses)
    res /= np.max(res)

    return res


if __name__ == '__main__':
    reco_space = odl.uniform_discr(min_pt=[-20, -20], max_pt=[20, 20], shape=[256, 256], dtype='float32')
    for _ in range(4):
        phantom = random_phantom(reco_space)
        print(phantom.shape)
        plt.imshow(phantom.asarray())
        plt.show()
        