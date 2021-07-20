import numpy as np
import torch
from matplotlib import pyplot as plt
import tqdm
from landfill.model import Landfill
from landfill.data import SingleImDataLoader
from scipy.ndimage import binary_dilation, binary_fill_holes

###############################################################################
# This is just a little example using the the torch implementation of         #
# Landfill with random gaussian fourier features.                             #
###############################################################################


n_features = 256 #Number of fourier features
n_layers = 8 #Number of hidden layers
hidden_width = 256 #Width of hidden layers
sigma=50. # Standard Dev. for frequencies of gaussian features
n_epochs = 30 # Training epochs
batch_size= 128# Batch size for 

im = np.load("./data/srs_image_1.npy")
mask = binary_fill_holes(binary_dilation(np.load("./data/mask_image_1.npy"), structure=np.ones((15,15))))

show_mask = np.zeros(im.shape+(4,))
show_mask[mask,-1] = 1

use_cuda = torch.cuda.is_available()

dataloader = SingleImDataLoader(im, mask, batch_size, cuda=torch.cuda.is_available())

lf = Landfill(sigma=sigma, n_fourier_features=n_features, n_layers=n_layers, hidden_width=hidden_width, coord_dims=2, other_dims=0) 
if torch.cuda.is_available():
    lf = lf.cuda()
    
ploss = torch.nn.PoissonNLLLoss(full=True, log_input=False)
opt = torch.optim.Adam(lf.parameters())

e_losses = []
e_std = []

# t_epoch = tqdm.tqdm(range(n_epochs))
# t_epoch.set_description("Epoch")

for epoch in range(n_epochs):
    losses = []

    t_train = tqdm.tqdm(dataloader, leave=False)
    t_train.desc = "Training batch"

    for X, Y in t_train:
        pred = lf(X)
        loss = ploss(pred,Y)
        losses.append(loss.clone().detach().item())
        opt.zero_grad()
        loss.backward()
        opt.step()
    e_losses.append(np.mean(losses))
    e_std.append(np.std(losses))
#     with torch.no_grad():
#         pred_im = lf(yx).numpy().reshape(N,N)
#     psnr = peak_signal_noise_ratio(srs_im, pred_im)
    #tqdm.write("\r")
    print(f"End of epoch {epoch+1}. Avg Loss = {e_losses[-1]:0.6f}.")

with torch.no_grad():
    yx = yx = torch.tensor(np.indices(im.shape).reshape(2,-1).T/im.shape[0]).to(torch.float) 
    if torch.cuda.is_available():
        yx = yx.cuda()
    pred_im = lf(yx).cpu().numpy().reshape(im.shape)

fig, ax = plt.subplots(1,3, figsize=(25,8))
ax[0].imshow(pred_im, vmin=0, vmax=255)
ax[0].set_title('Landfill Prediction')
ax[1].imshow(im)
ax[1].set_title('True Image')
ax[2].imshow(im)
ax[2].imshow(show_mask)
ax[2].set_title('Training Pixels')
plt.savefig("srs_landfill.png", dpi=200)

fig, ax = plt.subplots()
plt.errorbar(np.arange(n_epochs-1),e_losses[1:], yerr=e_std[1:])
plt.title("Loss after each training epoch (first omitted)")
plt.savefig("training_loss.png",dpi=200)
