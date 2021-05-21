import xarray as xr
import numpy as np
import tensorflow as tf
from tensorflow import keras as tfk
from tensorflow_probability import distributions as tfd
from tensorflow_probability import util as tfu
from matplotlib import pyplot as plt

###############################################################################
# This is just a little example using the the keras functional API and        #
# fourier features.                                                           #
###############################################################################


n_features = 100 #Number of fourier features
n_layers = 20 #Number of hidden layers
hidden_width = 3 #Width of hidden layers
sigma=30.

im = np.load("../data/srs_image_1.npy")
mask = np.load("../data/mask_image_1.npy")


#Encode fourier features
idx = np.indices(im.shape).reshape((2, -1)).T
V = idx/idx.max()
B = np.random.normal(scale=sigma, size=(2, n_features))
gamma = np.hstack((np.cos(2*np.pi*V@B), np.sin(2*np.pi*V@B)))

#Pixel values
y = im.flatten().astype(np.float32)

#Removed masked pixels
y = y[~mask.flatten()]
X = gamma[~mask.flatten()]

model = tfk.Sequential(
    [tfk.layers.Dense(n_features * 2, activation=tfk.layers.LeakyReLU())] + \
    [tfk.layers.Dense(hidden_width, activation=tfk.layers.LeakyReLU()) for i in range(n_layers)] + \
    [tfk.layers.Dense(1, activation='softplus')]
)

model.compile('adam', loss=tfk.losses.Poisson())
#This converges in 2 epochs
model.fit(X, y, epochs=2)


output = im.astype(np.float32)
output[mask] = model(gamma[mask.flatten()]).numpy().flatten()

background = model(gamma).numpy().reshape(im.shape)

plt.matshow(im)
plt.title("Input Image")

plt.matshow(output)
plt.title("In-filled Image")

plt.matshow(background)
plt.title("Background Estimate")

plt.matshow(im - background)
plt.title("Input Image - Background Estimate")

plt.show()
