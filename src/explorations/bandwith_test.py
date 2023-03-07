"""Example for reconstruction with FBP in 2d parallel geometry.

This example creates a filtered back-projection operator in 2d using the
ray transform and a ramp filter. This ramp filter is implemented in Fourier
space.

See https://en.wikipedia.org/wiki/Radon_transform#Inversion_formulas for
more information.

Also note that ODL has a utility function, `fbp_op` that can be used to
generate the FBP operator. This example is intended to show how the same
functionality could be implemented by hand in ODL.
"""

import numpy as np
import odl

from utils import analyticfilter
from utils import geometry
import torch

import matplotlib.pyplot as plt


# --- Set up geometry of the problem --- #


# Reconstruction space: discretized functions on the rectangle
# [-20, 20]^2 with 300 samples per dimension.
reco_space = odl.uniform_discr(
    min_pt=[-20, -20], max_pt=[20, 20], shape=[300, 300], dtype='float32')

# Angles: uniformly spaced, n = 1000, min = 0, max = pi
angle_partition = odl.uniform_partition(0, np.pi, 1000)

# Detector: uniformly sampled, n = 500, min = -30, max = 30
detector_partition = odl.uniform_partition(-30, 30, 500)

# Make a parallel beam geometry with flat detector
geometry1 = odl.tomo.Parallel2dGeometry(angle_partition, detector_partition)


# --- Create Filtered Back-projection (FBP) operator --- #


# Ray transform (= forward projection).
ray_trafo = odl.tomo.RayTransform(reco_space, geometry1)

# Fourier transform in detector direction
fourier = odl.trafos.FourierTransform(ray_trafo.range, axes=[1])

# Create ramp in the detector direction
ramp_function = fourier.range.element(lambda x: np.abs(x[1]) / (2 * np.pi))

# Create ramp filter via the convolution formula with fourier transforms
ramp_filter = fourier.inverse * ramp_function * fourier

# Create filtered back-projection by composing the back-projection (adjoint)
# with the ramp filter.
fbp = ray_trafo.adjoint * ramp_filter


# --- Show some examples --- #


# Create a discrete Shepp-Logan phantom (modified version)
phantom = odl.phantom.shepp_logan(reco_space, modified=True)

# Create projection data by calling the ray transform on the phantom
proj_data = ray_trafo(phantom)

# Calculate filtered back-projection of data
fbp_reconstruction = fbp(proj_data)

# Shows a slice of the phantom, projections, and reconstruction
#phantom.show(title='Phantom')
#proj_data.show(title='Projection Data (Sinogram)')
#fbp_reconstruction.show(title='Filtered Back-projection')
#(phantom - fbp_reconstruction).show(title='Error', force_show=True)



"addition"

geometry2=geometry.Geometry(angle_ratio=1,phi_size=1000,t_size=500)
geometry2.reco_space=reco_space

an_mod=analyticfilter.analytic_model(geometry2)



plt.cla()
plt.plot(an_mod.kernel_frequency_interval(), an_mod.kernel.detach().cpu(), label="filter in frequency domain")
plt.plot(ramp_function[3],label="filter ODL")
plt.legend()
plt.show()