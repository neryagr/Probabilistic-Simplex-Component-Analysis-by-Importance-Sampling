import torch.nn as nn
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class SISA(torch.nn.Module):
    def __init__(self, alpha, R=500, sigma=1e-2):
        super(SISA, self).__init__()
        self.sigma = torch.tensor(sigma, device=device, dtype=torch.float32)
        self.alpha = alpha
        self.R = R

    def forward(self, y, H, sigma=None):
        if sigma is None:
            sigma = self.sigma
        samp = torch.distributions.Dirichlet(self.alpha).rsample((self.R,)).type(torch.float32).to(device)
        yzSy = torch.einsum("ij,hj->hi", H, samp) - y[:, None, :]

        pyz = torch.einsum("...ij,...ij->...i", yzSy, yzSy) / (-2 * sigma)

        weights = torch.nn.Softmax(dim=-1)(pyz)
        zy, zzy = torch.einsum("j...k,...j->...k", samp, weights), torch.einsum("j...k,j...l,...j->...kl", samp, samp,
                                                                                weights)
        return zy, zzy


class LISA(torch.nn.Module):
    def __init__(self, alpha, R=500, sigma=1e-2):
        super(LISA, self).__init__()
        self.sigma = sigma
        self.R = R
        self.alpha = torch.clone(alpha)
        self.reg = 0.8

    def forward(self, y, H, sigma=None):
        if sigma is None:
            sigma = self.sigma
        alpha0 = self.alpha.sum()
        m = self.alpha / alpha0
        C = (torch.diag_embed(m) - torch.outer(m, m)) / (alpha0 + 1)

        EY = torch.einsum('dr,r->d', H, m)
        Cyy = torch.einsum('dr,rk,tk->dt', H, C, H) + sigma * torch.eye(H.shape[0], device=device, dtype=torch.float32)
        invCyy = torch.linalg.inv(Cyy)
        LMMSE_m = nn.functional.relu(m + torch.einsum('rk, dk, dt, ...t -> ...r', C, H, invCyy, y - EY)) + 1e-6
        LMMSE_m = (LMMSE_m.T / LMMSE_m.sum(axis=1)).T
        LMMSE_C = C - torch.einsum('rk, dk, dt, tp, pl -> rl', C, H, invCyy, H, C)
        trace_LMMSE_C = nn.functional.relu(torch.einsum("ii->", LMMSE_C)) + 1e-6
        k = (1 - (LMMSE_m ** 2).sum(axis=-1)) / trace_LMMSE_C - 1
        diri = nn.functional.relu((k * LMMSE_m.T).T) + self.reg

        samp = torch.distributions.Dirichlet(diri).rsample((self.R,)).type(torch.float32).to(device)
        yzSy = torch.einsum("...ij,h...j->...hi", H, samp) - y[:, None, :]
        pyz = torch.einsum("...ij,...ij->...i", yzSy, yzSy) / (-2 * sigma)
        qz = torch.distributions.Dirichlet(diri).log_prob(samp).T
        pz = torch.distributions.Dirichlet(self.alpha).log_prob(samp).T
        weights = torch.nn.Softmax(dim=-1)(pyz + pz - qz)

        zy, zzy = torch.einsum("j...k,...j->...k", samp, weights), torch.einsum("j...k,j...l,...j->...kl", samp, samp,
                                                                                weights)
        return zy, zzy
