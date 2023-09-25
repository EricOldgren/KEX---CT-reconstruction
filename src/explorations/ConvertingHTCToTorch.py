import numpy as np
import scipy.io
import torch
# import cv2
import matplotlib.pyplot as plt

Uw = 20

def convertHTCSinograms():
    HTCTrainingFileNames = ["htc2022_solid_disc_full.mat", "htc2022_ta_full.mat", "htc2022_tb_full.mat", "htc2022_tc_full.mat", "htc2022_td_full.mat"]

    HTCTrainingData = []

    for elem in HTCTrainingFileNames:
        data = scipy.io.loadmat('data\HTC2022\TrainingData\\'+elem)
        cellData = data['CtDataFull'][0,0]['sinogram']
        HTCTrainingData.append(torch.tensor(cellData))

    torch.save(HTCTrainingData,"data\HTC2022\HTCTrainingData.pt")

    LoadedData = torch.load("data\HTC2022\HTCTrainingData.pt")

    HTCTestDataLimited = []
    HTCTestDataFull = []

    for i in ['a','b','c']:
        for j in range(1,8):
            dataFull = scipy.io.loadmat('data\HTC2022\TestData\htc2022_0' + str(j) + i + '_full.mat')
            dataLimited = scipy.io.loadmat('data\HTC2022\TestData\htc2022_0' + str(j) + i + '_limited.mat')
            cellDataFull = dataFull['CtDataFull'][0,0]['sinogram']
            cellDataLimited = dataLimited['CtDataLimited'][0,0]['sinogram']
            HTCTestDataFull.append(torch.tensor(cellDataFull))
            HTCTestDataLimited.append(torch.tensor(cellDataLimited))

    torch.save(HTCTestDataFull,'data\HTC2022\HTCTestDataFull.pt')
    torch.save(HTCTestDataLimited,'data\HTC2022\HTCTestDataLimited.pt')

def convertHTCPhantoms(path_ending="_seg"):
    HTCTrainingPhantoms = []
    HTCTestPhantomsFull = []
    HTCTestPhantomsLimited = []

    for i in ['solid_disc', 'ta', 'tb', 'tc', 'td']:
        data = scipy.io.loadmat('data\HTC2022\TrainingData\htc2022_' + i + '_full_recon_fbp' + path_ending + '.mat')
        cellData = data['reconFullFbp'+("Seg" if path_ending else "")]
        HTCTrainingPhantoms.append(torch.tensor(cellData))

    for i in ['a','b','c']:
        for j in range(1,8):
            dataFull = scipy.io.loadmat('data\HTC2022\TestData\htc2022_0' + str(j) + i + '_recon_fbp' + path_ending + '.mat')
            dataLimited = scipy.io.loadmat('data\HTC2022\TestData\htc2022_0' + str(j) + i + '_recon_fbp' + path_ending + '_limited.mat')
            cellDataFull = dataFull['reconFullFbp' + ("Seg" if path_ending else "")]
            cellDataLimited = dataLimited['reconLimitedFbp' + ("Seg" if path_ending else "")]
            HTCTestPhantomsFull.append(torch.tensor(cellDataFull))
            HTCTestPhantomsLimited.append(torch.tensor(cellDataLimited))
    
    return torch.stack(HTCTrainingPhantoms), torch.stack(HTCTestPhantomsFull), torch.stack(HTCTestPhantomsLimited)
    # torch.save(HTCTrainingPhantoms, 'data\HTC2022\HTCTrainingPhantoms.pt')
    # torch.save(HTCTestPhantomsFull, 'data\HTC2022\HTCTestPhantomsFull.pt')
    # torch.save(HTCTestPhantomsLimited, 'data\HTC2022\HTCTestPhantomsLimited.pt')
    #Images are 512x512

def convert2DeteCT(mapPathInitial='data\\2DeteCT_slices1-1000_RecSeg', mapPathTarget='data\\2DeteCT_slices1-1000Mode2Rec'):
    for i in range(1,1001):
        index = str(i).zfill(5)
        location = mapPathInitial  + '\\slice' + index + '\\mode2\\reconstruction.tif'
        currentSlice = torch.load(location)
        torch.save(currentSlice,mapPathTarget + '\slice' + index)


if __name__ == "__main__":
    from geometries import HTC2022_GEOMETRY, DEVICE
    from utils.tools import GIT_ROOT, MSE, segment_imgs, htc_score
    import matplotlib
    matplotlib.use("WebAgg")
    import matplotlib.pyplot as plt
    import json
    train_seg, test_seg, testlimited_seg = convertHTCPhantoms(path_ending="_seg")
    train, test, testlimited = convertHTCPhantoms(path_ending="")
    sinos = torch.stack(torch.load(GIT_ROOT / "data/HTC2022/HTCTrainingData.pt", map_location=DEVICE))[:, :720]
    print("sinos loaded")
    sinos = HTC2022_GEOMETRY.rotate_sinos(sinos, 180)
    print(sinos.shape)
    recons = (HTC2022_GEOMETRY.fbp_reconstruct(sinos))
    print("beginning segmentation...")
    recon_seg = recons > 0.02 # segment_imgs(torch.nn.functional.relu(recons))


    numeric = HTC2022_GEOMETRY.project_forward(recons)

    print("reprojected noise:", MSE(numeric, sinos))

    disp_ind = 3
    plt.imshow(train[disp_ind].cpu())
    plt.colorbar()
    plt.figure()
    plt.imshow(recons[disp_ind].cpu())
    plt.title("recon")
    plt.colorbar()
    plt.figure()
    plt.imshow(numeric[disp_ind].cpu())
    plt.colorbar()
    plt.title("numeric")
    plt.figure()
    plt.imshow(sinos[disp_ind].cpu())
    plt.colorbar()
    plt.title("real")

    plt.figure()
    plt.imshow((sinos-numeric)[disp_ind].cpu())
    plt.colorbar()
    plt.title("diff")

    plt.figure()
    plt.imshow(recon_seg[disp_ind].cpu())
    plt.title("seg recon")

    print("HTCscore:", htc_score(recon_seg, train_seg.to(torch.bool)))
    print("HTCscore:", htc_score(recons > 0.03, train_seg.to(torch.bool)))
    print("HTCscore:", htc_score(recons > 0.02, train_seg.to(torch.bool)))
    print("HTCscore:", htc_score(recons > 0.016, train_seg.to(torch.bool)))
    print("HTCscore:", htc_score(recons > 0.01, train_seg.to(torch.bool)))


    mean_mu = recons[train_seg].mean()
    var_mu = recons[train_seg].var()
    noise_lvl = MSE(numeric, sinos)

    (GIT_ROOT / "data/HTC2022/conclusions.json").write_text(
        json.dumps({
            "attenuation_mean": mean_mu.item(),
            "attenuation_var": var_mu.item(),
            "sino_noise":noise_lvl.item(),
            "disc_radius":35,
            "space_width":38
        })   
    )


    plt.show()
    # convertHTCSinograms()



    print("done")