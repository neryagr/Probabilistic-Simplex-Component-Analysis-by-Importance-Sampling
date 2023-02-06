import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def matrix_sample(m, d, box=[0, 1]):
    H = torch.rand((m, d), dtype=torch.float32, device=device) * (box[1] - box[0]) + box[0]
    return H


def latent_sample(n, d, alpha=None):
    if alpha is None:
        alpha = torch.ones(d)
    Z = torch.distributions.Dirichlet(alpha).rsample((n,)).type(torch.float32).to(device)
    return Z


def noise_sample(H, z, sigma, snr):
    n = z.shape[0]
    m = H.shape[0]
    noise = torch.randn((n, m), dtype=torch.float32, device=device)
    if snr is not None:
        mult = (((torch.einsum("ij,nj", H, z) ** 2).mean() / (10 ** (snr / 10)))).item()
    else:
        mult = sigma
    return noise * (mult ** 0.5), mult


def sample(n, m, d=10, sigma=1e-3, snr=None, alpha=None):
    H = matrix_sample(m, d)
    z = latent_sample(n, d, alpha)
    noise, sigma = noise_sample(H, z, sigma, snr)
    Y = torch.einsum("ij,nj->ni", H, z) + noise

    return Y, H, z, sigma


def resample(H, n, sigma=1e-3, snr=None, alpha=None):
    m, d = H.shape
    z = latent_sample(n, d, alpha)
    noise, sigma = noise_sample(H, z, sigma, snr)
    Y = torch.einsum("ij,nj->ni", H, z) + noise
    return Y, z, sigma
