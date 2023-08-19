import torch
from utils.polynomials import no_bdry_linspace, linear_upsample_no_bdry, down_sample_no_bdry
import random


def test_x_scale():
    
    for _ in range(100):
        n_points = random.randint(10, 1000)
        factor = random.randint(10,50)*2+1
        start = random.normalvariate(0,1)*10-5
        L = random.normalvariate(0,1)*10
        end = start + L
        xs = no_bdry_linspace(start, end, n_points)
        xs_full = no_bdry_linspace(start, end, n_points*factor)
        xs_upsampled = linear_upsample_no_bdry(xs, factor)
        # print(torch.linalg.norm(xs_full-xs_upsampled).item())
        # print((torch.mean(torch.abs(xs_full-xs_upsampled)) / torch.finfo(torch.float).eps).item())
        # print(torch.mean((xs_full-xs_upsampled)**2).item())
        assert torch.mean((xs_full-xs_upsampled)**2) < 1e-6
    
    print(("x scale upsampling ok!"))

def test_y_scale():
    for _ in range(100):
        n_points = random.randint(10, 1000)
        factor = random.randint(10,50)*2+1 #must be odd

        f = torch.randn(n_points)
        f[:2] = 0
        f[-2:] = 0
        integral1 = torch.sum(f)
        f_upscaled = linear_upsample_no_bdry(f, factor)
        integral2 = torch.sum(f_upscaled) / factor

        # print((integral1-integral2)**2)
        assert (integral1-integral2)**2 < 1e-6
    
    print("y scale upsampling ok!")

def test_downsapling():

    for _ in range(100):
        n_points = random.randint(10, 1000)
        factor = random.randint(10,50)*2+1 #must be odd

        f = torch.randn(n_points)
        f_upscaled = linear_upsample_no_bdry(f, factor)
        up_down = down_sample_no_bdry(f_upscaled, factor)
        assert (f == up_down).all()
    
    print("down sampling ok!")

if __name__ == "__main__":
    test_x_scale()
    test_y_scale()
    test_downsapling()
