import pickle
from utils.geometry import ParallelGeometry, setup

with open("test_pickle.pt", "rb") as file:
    model = pickle.load(file)

print(model)

g = model.geometry
_, _, test_sinos, test_y = setup(g, num_to_generate=10)

# train_sinos, train_y, test_sinos, test_y = setup(geometry, num_to_generate=0, train_ratio=0.5, use_realistic=True, data_path="data/kits_phantoms_256.pt")

model.visualize_output(test_sinos, test_y, output_location="show")

print("done")