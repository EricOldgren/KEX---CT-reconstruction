import numpy as np
import odl

from odl.phantom import ellipsoid_phantom, shepp_logan_ellipsoids
from odl import DiscretizedSpace, DiscretizedSpaceElement
import matplotlib.pyplot as plt

def ellipsoid_structure(value: float, x_radius: float, y_radius: float, x_center: float, y_center: float, rotation: float):
    return [value, x_radius, y_radius, x_center, y_center, rotation]

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

def unstructured_random_phantom(reco_space: DiscretizedSpace, num_ellipses = 10):
    "Phantom of random ellipses. More random and doesn't look like a brain. Maybe useful for more diverse data."

    ellipsoids = []
    for _ in range(num_ellipses):
        ellipsoids.append(random_ellipsoid([-1.0, -1.0], [1.0, 1.0], value = np.random.uniform(0.1, 0.6)))

    res = ellipsoid_phantom(reco_space, ellipsoids)
    res /= np.max(res)

    return res

def random_phantom(reco_space: DiscretizedSpace, num_ellipses = 10)->DiscretizedSpaceElement:

    brain_center = np.random.uniform(-0.1, 0.1, 2)
    brain_r = np.random.uniform(0.5, 0.7, 2)
    min_s = np.min(brain_r) / np.sqrt(2)
    skull = [1.0, brain_r[0], brain_r[1], brain_center[0], brain_center[1], 0.0]
    brain = [-0.5, brain_r[0]*0.95, brain_r[1]*0.95, brain_center[0], brain_center[1], 0.0]

    ellipses = [skull, brain]
    for _ in range(num_ellipses):
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

        unstructured = unstructured_random_phantom(reco_space)
        plt.imshow(unstructured)
        plt.show()
        