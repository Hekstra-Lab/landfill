__all__ = [
    "fourier_encode",
    "prep_data",
    "create_model",
    "backfill",
]
import numpy as np
from tensorflow import keras as tfk
from matplotlib import pyplot as plt

###############################################################################
# This is just a little example using the the keras functional API and        #
# fourier features.                                                           #
###############################################################################


# n_features = 100 #Number of fourier features
# sigma=30.


# Encode fourier features
def fourier_encode(im, sigma, n_features):
    """
    Parameters
    ----------
    im : (M, N) array-like
    sigma : float
        The scale of the Gaussian from which the weights are drawn
    n_features : int
        How many fourier features to sample

    Returns
    -------
    gamma : () array-like
        The encoded image
    """
    idx = np.indices(im.shape).reshape((2, -1)).T
    V = idx / idx.max()
    B = np.random.normal(scale=sigma, size=(2, n_features))
    return np.hstack((np.cos(2 * np.pi * V @ B), np.sin(2 * np.pi * V @ B)))


def prep_data(im, mask, sigma, n_features=20):
    """
    Parameters
    ----------
    im : (M, N) array-like
    mask : (M, N) array-like of bool
    sigma : float
        The scale of the Gaussian from which the weights are drawn
    n_features : int, default: 20

    Returns
    -------
    X, y : array-like
        The training data for the model
    gamma
        The encoded image
    """
    gamma = fourier_encode(im, sigma=sigma, n_features=n_features)
    y = im.flatten().astype(np.float32)

    # Removed masked pixels
    y = y[~mask.flatten()]
    X = gamma[~mask.flatten()].astype(np.float32)
    return X, y, gamma


def create_model(
    n_features, n_layers=20, hidden_width=3, loss=tfk.losses.Poisson(), optimizer="adam"
):
    """
    Convenience function for creating a Keras perceptron

    Parameters
    ----------
    n_features : int
    n_layers : int, default: 20
        The number of hidden layers
    hidden_width : int, default: 3
        The width of the hidden layers
    loss : tf loss function
    optimizer : str, default: 'adam'

    Returns
    -------
    keras.Sequential
    """
    model = tfk.Sequential(
        [tfk.layers.Dense(n_features * 2, activation=tfk.layers.LeakyReLU())]
        + [
            tfk.layers.Dense(hidden_width, activation=tfk.layers.LeakyReLU())
            for i in range(n_layers)
        ]
        + [tfk.layers.Dense(1, activation="softplus")]
    )
    model.compile(optimizer, loss=loss)
    return model


def backfill(im, mask, gamma, model):
    """
    Fill in masked areas of an image using the model.

    Parameters
    ----------
    im : (M, N) array-like
    mask : (M, N) array-like of bool
    gamma : (n_features, 2) TODO get these numbers right
    model : keras model

    Returns
    -------
    output, background : (M, N) array-like
        The in-filled image and the background estimate
    """

    output = im.astype(np.float32)
    output[mask] = model(gamma[mask.flatten()]).numpy().flatten()

    background = model(gamma).numpy().reshape(im.shape)
    return output, background


def visualize(im, output, background, share_norm=True, sharex=True, sharey=True):
    """
    Parameters
    ----------
    im : (M, N) array-like
        The original image
    output : (M, N) array-like
        The in-filled image
    background :
        The estimated background
    share_norm : bool
        Whether to share the norm between the subplots

    Returns
    -------
    fig : matplotlib.figure
    axs : (3,) axes
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 3, sharex=sharex, sharey=sharey)
    if share_norm:
        vmax = np.max(np.maximum.reduce([im, output, background]))
        vmin = np.min(np.minimum.reduce([im, output, background]))
    else:
        vmax = None
        vmin = None
    axs[0].matshow(im, vmin=vmin, vmax=vmax)
    axs[0].set_title("Input Image")

    axs[1].matshow(output, vmin=vmin, vmax=vmax)
    axs[1].set_title("In-filled Image")

    axs[2].matshow(background, vmin=vmin, vmax=vmax)
    ax.set_title("Background Estimate")
    return fig, axs
