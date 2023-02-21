import odl
from src.utils.geometry import Geometry
from src.utils.data_generator import unstructured_random_phantom
geometry = Geometry(1.0, 100, 300)

ft_op = odl.trafos.FourierTransform(geometry.reco_space)
print(ft_op.domain, ft_op.range)
phantom = unstructured_random_phantom(geometry.reco_space)
phantom.show()

ft_phantom = ft_op(phantom)
print(ft_phantom.shape)

ft_phantom.show(force_show=True)