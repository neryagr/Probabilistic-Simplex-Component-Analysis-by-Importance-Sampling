from problem.sample import sample, resample
from EM.EM import *
from EM.ISA import *
from base_lines.VCA import vca
from problem.errors import *
from base_lines.VIA import VIA_GD
import matplotlib.pyplot as plt
print("run")

trails=50
m = 50
d = 20
ns=[1000,2000,3000,4000,5000,6000,8000,10000]
snr = 20
sigma=1e-2
alpha=torch.ones(d)
R = 500

algosNames=["VCA","VIA","SISA","LISA"]
errors_EM = np.zeros((trails,len(algosNames),len(ns)))
pbar=range(trails)

samples=10000

for t in pbar:
    y, H, z, sigma = sample(samples, m, d, sigma, snr=snr)
    eSISA = SISA(alpha,R,sigma)
    eLISA = LISA(alpha,R,sigma)
    for i,n in enumerate(ns):
        Y=y[:n,:]

        j=0
        Hvca, _, _ = vca(Y.cpu().numpy().T, d)
        errors_EM[t, j, i] = error(H.cpu().numpy(), Hvca)
        j+=1
        Hvca = torch.tensor(Hvca, dtype=torch.float32, device=device)


        Hsisa = EM(Y, d, eSISA, H=torch.clone(Hvca),sigma=sigma)


        Hvia=VIA_GD(Y,torch.tensor(Hsisa, dtype=torch.float32, device=device),sigma)
        errors_EM[t, j, i] = error(H.cpu().numpy(), Hvia)
        j+=1


        errors_EM[t, j, i] = error(H.cpu().numpy(), Hsisa)
        j+=1
        Hlisa = EM(Y, d, eLISA, H=torch.clone(Hvca),sigma=sigma)
        errors_EM[t, j, i] = error(H.cpu().numpy(), Hlisa)
        j+=1
for j,name in enumerate(algosNames):
    plt.plot(ns, 10*np.log10(errors_EM[:,j,:].mean(axis=-2)))
plt.legend(algosNames)
plt.xlabel("n")
plt.ylabel("MSE[dB]")
plt.grid()
plt.show()
print("finish")
