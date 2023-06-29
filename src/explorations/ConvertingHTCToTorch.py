import numpy as np
import scipy.io
import torch

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


