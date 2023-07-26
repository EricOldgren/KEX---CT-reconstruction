import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from utils.parallel_geometry import ParallelGeometry, setup
from models.fbpnet import FBPNet
from models.fouriernet import FNO_BP
from models.fbps import FBP

ANGLE_RATIOS = [0.25]# 0.8, 0.85, 0.9, 0.95, 1.0]
EPOPCHS =      [100]# 100, 100, 100,  100, 60]
TRAINED = {}
LAMBDA  = 10 #regularization parameter

for ar, n_epochs in zip(ANGLE_RATIOS, EPOPCHS):
    geometry = ParallelGeometry(ar, 450, 300) #50,40

    (train_sinos, train_y, test_sinos, test_y) = setup(geometry, num_to_generate=1500,use_realistic=True, pre_computed_phantoms=torch.load("data\constructed_data.pt"),data_path="data/kits_phantoms_256.pt") #, pre_computed_phantoms=torch.load("data\constructed_data.pt")
    model = FBPNet(geometry,n_fbps=4,use_padding=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = lambda diff : torch.mean(diff*diff)

    dataloader = DataLoader(list(zip(train_sinos, train_y)), batch_size=20, shuffle=True)

    for epoch in range(n_epochs):
        #if epoch % 10 == 0:
            #model.visualize_output(test_sinos, test_y, loss_fn, output_location="show")
        for sinos, y in dataloader:
            out = model(sinos)

            loss = loss_fn(out - y) 
            loss += model.regularization_term()*LAMBDA
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(f"epoch {epoch} loss: {loss.item()}")
    #model.visualize_output(test_sinos, test_y, loss_fn, output_location="show")
    
    TRAINED[ar] = model

torch.save(TRAINED[0.25].state_dict(), "results\prev_res.pt")

