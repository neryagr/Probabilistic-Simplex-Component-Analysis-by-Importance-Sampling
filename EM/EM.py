import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def EM(Y, d, eISA, H=None, sigma=1e-3, tol=1e-5, maxRounds=100):
    n, m = Y.shape
    l = [2, 1]
    if H is None:
        H = torch.rand((m, d), dtype=torch.float32, device=device)
    i = 0
    while abs(l[-1] - l[-2]) > tol and i < maxRounds:
        i += 1

        ez,ezz = eISA.forward(Y, H, sigma)

        NLL = -(2 * torch.einsum("ij,jk,ik", Y, H, ez) - torch.einsum("ij,ik,nkj->", H, H, ezz)) / Y.shape[0]
        l.append(NLL.item())

        H = torch.einsum("ij,ik,kl->jl", Y, ez, torch.linalg.pinv(ezz.sum(axis=0)))
    return H.cpu().numpy()
