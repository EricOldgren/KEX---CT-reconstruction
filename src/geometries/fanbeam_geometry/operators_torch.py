import torch
"""
Attempt to implement all operators in pytorch.
"""


torch.jit.enable_onednn_fusion(True)
@torch.jit.script
def _project_forward(data: torch.Tensor, Xs: torch.Tensor, Ys: torch.Tensor, betas: torch.Tensor, us: torch.Tensor, R: float, DEVICE: torch.device, interpolation_method: int = 0, tensor_batch_size: int = 1):
    """
        Fanbeam forward projection
    """

    data = torch.flip(data, dims=(1,)) #flip y upwards
    
    N_samples, _, _ = data.shape
    Xs, Ys = Xs.reshape(-1), Ys.reshape(-1)
    betas, us = betas.reshape(-1, 1), us.reshape(1, -1)
    Nb, _ = betas.shape
    _, Nu = us.shape
    Xmin, Xmax, Ymin, Ymax = Xs[0], Xs[-1], Ys[0], Ys[-1]
    dX, dY = torch.mean(Xs[1:]-Xs[:-1]), torch.mean(Ys[1:] - Ys[:-1])
    min_d = torch.minimum(dX, dY)

    N_line_points = int(((Xmax - Xmin)**2 + (Ymax-Ymin)**2)**0.5 / min_d)
    ratios = (1/(2*N_line_points) + torch.arange(0,N_line_points, device=DEVICE, dtype=torch.float32)/N_line_points)

    # Define lines of Rays
    Sources = torch.stack([torch.cos(betas), torch.sin(betas)],dim=-1).reshape(-1,1,2)*R
    ProjectorPoints = torch.stack([torch.sin(betas), -torch.cos(betas)], dim=-1).reshape(-1,1,2) * us.reshape(1,-1,1)
    line_directions = ProjectorPoints - Sources
    line_directions /= torch.linalg.norm(line_directions)
    line_normals = torch.stack([line_directions[:, :, 1], -line_directions[:, :, 0]], dim=-1)
    line_translations = torch.sum(ProjectorPoints*line_normals, dim=-1, keepdim=True)
    
    # Find endpoints of lines in region
    bounding_normals = torch.tensor([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]], device=DEVICE, dtype=torch.float32) #edges
    bounding_vals = torch.stack([Xmin, -Xmax, Ymin, -Ymax])
    A = torch.stack([
        bounding_normals[None, None].repeat(Nb, Nu, 1, 1), line_normals[:,:,None].repeat(1,1,4,1)
    ], dim=3)
    ts = torch.stack([
        bounding_vals[None, None, :, None].repeat(Nb, Nu, 1, 1), line_translations[:,:,None].repeat(1,1,4,1)
    ], dim=3)
    intersections: torch.Tensor = torch.linalg.solve(A, ts).reshape(Nb, Nu, 4, 2) #Solve for intersection of rays and edges
    on_inside = ((torch.einsum("nk,buik->buin", bounding_normals, intersections) - bounding_vals) > -1e-5).sum(dim=-1) == 4
    invalids = (on_inside.sum(dim=-1) != 2)
    on_inside[invalids, :] = torch.tensor([1,1,0,0], device=DEVICE, dtype=torch.bool)
    intersections = intersections[on_inside].view(Nb, Nu, 2, 2)
    #Define line points to integrate over
    start_points, stop_points = intersections[:, :, 0]*0, intersections[:,:,0]*0
    zero2one = torch.sum((intersections[:,:, 1]-intersections[:,:,0])*line_directions, dim=-1) > 0.0 #orientations where line_dir goes from int[0] - int[1]
    start_points[zero2one, :] += intersections[zero2one][:, 0]; stop_points[zero2one, :] += intersections[zero2one][:, 1]
    start_points[~zero2one, :] += intersections[~zero2one][:, 1]; stop_points[~zero2one, :] += intersections[~zero2one][:, 0]
    line_segments = stop_points - start_points
    dls = torch.linalg.norm(line_segments, dim=-1, keepdim=True)
    dls /= N_line_points
    dls[invalids] *= 0

    line_points: torch.Tensor = ratios.reshape(1,1,-1,1)*line_segments[...,None, :]
    line_points += start_points[...,None, :]
    line_points[invalids] *= 0
    
    res = torch.zeros((N_samples, Nb, Nu), device=DEVICE, dtype=betas.dtype)
    
    if interpolation_method == 0: #Nearest Neighbour
        #Find inds of line points
        line_points -= torch.stack([Xmin, Ymin])
        line_points /= torch.stack([dX, dY])
        line_points += 0.5
        inds = line_points.to(torch.int64) #integer inds

        out_mul = torch.zeros((tensor_batch_size, Nb, Nu, N_line_points), device=DEVICE, dtype=betas.dtype)
        out_sum = torch.zeros((tensor_batch_size, Nb, Nu), device=DEVICE, dtype=betas.dtype)

        for ind in range(0, N_samples, tensor_batch_size):
            nxt = min(N_samples, ind+tensor_batch_size)
            if nxt-ind < tensor_batch_size:
                out_mul.resize_((nxt-ind, Nb, Nu, N_line_points))
                out_sum.resize_((nxt-ind, Nb, Nu))

            torch.mul(data[ind:nxt, inds[...,1], inds[...,0]], dls, out=out_mul)
            torch.sum(out_mul, dim=-1, out=out_sum)
            res[ind:nxt] += out_sum
            # res[ind] += torch.einsum("ubl,ubl->ub", data[ind, inds[...,1], inds[...,0]], dls) # this uses nearest neighbour interpolation - faster

    if interpolation_method == 1: #Bilinear
        data = torch.nn.functional.pad(data, (0,1,0,1))
        line_points -= torch.stack([Xmin, Ymin])
        line_points /= torch.stack([dX, dY])
        inds = line_points.to(torch.int64)
        ratios = torch.frac(line_points)

        res = torch.zeros((N_samples, Nb, Nu), device=DEVICE, dtype=betas.dtype)

        for ind in range(N_samples):
            res[ind] += torch.einsum("ubl,ubl,ubl,ubl->ub", data[ind, 1:, 1:][inds[...,1], inds[...,0]], dls, ratios[...,0], ratios[...,1]) #xy
            ratios[...,0] *= -1
            ratios[...,0] += 1
            res[ind] += torch.einsum("ubl,ubl,ubl,ubl->ub", data[ind, :-1, 1:][inds[...,1], inds[...,0]], dls, ratios[...,0], ratios[...,1]) #(1-x)y
            ratios[...,1] *= -1
            ratios[...,1] += 1
            res[ind] += torch.einsum("ubl,ubl,ubl,ubl->ub", data[ind, :-1, :-1][inds[...,1], inds[...,0]], dls, ratios[...,0], ratios[...,1]) #(1-x)(1-y)
            ratios[...,0] *= -1
            ratios[...,0] += 1
            res[ind] += torch.einsum("ubl,ubl,ubl,ubl->ub", data[ind, 1:, :-1][inds[...,1], inds[...,0]], dls, ratios[...,0], ratios[...,1]) #x(1-y)
            ratios[...,1] *= -1
            ratios[...,1] += 1
            #xy
    
    return res

def _project_backwarde(*args):
    raise NotImplementedError()
 
    