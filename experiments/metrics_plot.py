import numpy as np
import matplotlib.pyplot as plt

psnr = np.load("./sigma_sweep/psnr.npy")
loss = np.load("./sigma_sweep/test_loss.npy")
sigma = np.logspace(1,9,25, base=2)

plt.figure()
plt.plot(sigma, 4*loss, label = '4 x Poisson NLL (Lower is bettter)')
plt.plot(sigma, psnr, label='PSNR (Higher is better)')
plt.title("Hyperparameter Sweep", fontsize=16)
plt.semilogx()
plt.xlabel('Sigma')
plt.legend()
plt.tight_layout()
plt.savefig("./data/metrics.png")

