import numpy as np
import torch
from matplotlib import pyplot as plt
import tqdm
from landfill.model import Landfill
from landfill.data import SingleImDataLoader
from scipy.ndimage import binary_dilation, binary_fill_holes
from skimage.data import camera
from skimage.metrics import peak_signal_noise_ratio

###############################################################################
# This is just a little example using the the torch implementation of         #
# Landfill with random gaussian fourier features.                             #
###############################################################################


n_features = 256 #Number of fourier features
n_layers = 8 #Number of hidden layers
hidden_width = 256 #Width of hidden layers
#sigma=50. # Standard Dev. for frequencies of gaussian features
n_epochs = 30 # Training epochs
batch_size= 128# Batch size for 

im = camera()
pix = torch.tensor(im.reshape(-1)).to(torch.float)
if torch.cuda.is_available():
    pix = pix.cuda()

mask = np.random.choice([False, True], p=[0.75, 0.25], size=im.shape)

show_mask = np.zeros(im.shape+(4,))
show_mask[mask,-1] = 1

yx = yx = torch.tensor(np.indices(im.shape).reshape(2,-1).T/im.shape[0]).to(torch.float) 
if torch.cuda.is_available():
    yx = yx.cuda()

dataloader = SingleImDataLoader(im, mask, batch_size, cuda=torch.cuda.is_available())
psnrs = []
test_losses = []

for i, sigma in enumerate(np.logspace(1, 9, 25, base=2)):

    lf = Landfill(sigma=sigma, n_fourier_features=n_features, n_layers=n_layers, hidden_width=hidden_width, coord_dims=2, other_dims=0) 
    if torch.cuda.is_available():
        lf = lf.cuda()
        
    ploss = torch.nn.PoissonNLLLoss(full=True, log_input=False)
    opt = torch.optim.Adam(lf.parameters())

    e_losses = []
    e_std = []

    for epoch in range(n_epochs):
        losses = []

        # t_train = tqdm.tqdm(dataloader, leave=False)
        # t_train.desc = f"Sigma ={sigma:.3f} - Epoch {epoch}/{n_epochs}"

        for X, Y in dataloader:
            pred = lf(X)
            loss = ploss(pred,Y)
            losses.append(loss.clone().detach().item())
            opt.zero_grad()
            loss.backward()
            opt.step()
        e_losses.append(np.mean(losses))
        e_std.append(np.std(losses))
        #print(f"End of epoch {epoch+1}. Avg Loss = {e_losses[-1]:0.6f}.")

    with torch.no_grad():
        lf = lf.eval()
        pred_pix = lf(yx)
        test_losses.append(ploss(pred_pix, pix).clone().detach().item())

    pred_im = pred_pix.cpu().numpy().reshape(im.shape)
    psnrs.append(peak_signal_noise_ratio(im, pred_im))
    
    print(f"Sigma = {sigma} -- Loss = {test_losses[-1]:0.6f} -- PSNR = {psnrs[-1]:0.4f}")

    fig, ax = plt.subplots(1,3, figsize=(25,8))
    ax[0].imshow(pred_im, vmin=0, vmax=255)
    ax[0].set_title('Landfill Prediction')
    ax[1].imshow(im)
    ax[1].set_title('True Image')
    ax[2].imshow(im)
    ax[2].imshow(show_mask)
    ax[2].set_title('Training Pixels')
    plt.savefig(f"sigma_sweep/srs_landfill_{i}.png", dpi=200)

    fig, ax = plt.subplots()
    plt.errorbar(np.arange(n_epochs-1),e_losses[1:], yerr=e_std[1:])
    plt.title("Loss after each training epoch (first omitted)")
    plt.savefig(f"sigma_sweep/training_loss_{i}.png",dpi=200)

np.save("sigma_sweep/psnr.npy", np.array(psnrs))
np.save("sigma_sweep/test_loss.npy", np.array(test_losses))
