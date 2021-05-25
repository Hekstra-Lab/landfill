from landfill.landfill import create_model
import numpy as np
import landfill as lf

im = np.load("../data/srs_image_1.npy")
mask = np.load("../data/mask_image_1.npy")

n_features = 20
X, y, gamma = lf.prep_data(im, mask, n_features)
print(X.dtype)
print(y.dtype)
model = create_model(n_features)

model.fit(X.astype(np.float32), y, epochs=2)
from IPython import embed

embed()
