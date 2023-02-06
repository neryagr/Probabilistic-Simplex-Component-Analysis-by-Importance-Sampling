import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def gradient_descent_with_backtracking_step(fcn, x, backtracking_alpha=0.5, backtracking_beta=0.5, max_step_size=1e2,
                                            min_step_size=1e-9, boxd=[1e-3, 1e3]):
    x.requires_grad_(True)
    f = fcn(x)
    f.backward()
    with torch.no_grad():
        grad = x.grad
        sqrd_grad = torch.sum(grad ** 2)
        step_size = max_step_size
        x_new = x - step_size * grad
        x_new.clamp_(boxd[0], boxd[1])
        new_f = fcn(x_new)

        while (step_size > min_step_size and new_f > (f - backtracking_alpha * step_size * sqrd_grad)):
            step_size = backtracking_beta * step_size
            x_new = x - step_size * grad
            x_new.clamp_(boxd[0], boxd[1])
            new_f = fcn(x_new)
        if step_size > min_step_size:
            x = x_new
            success = True
        else:
            success = False
    x.requires_grad_(False)
    return x


def min_func(Y, H, diri, sigma):
    mean = Y - torch.einsum("ij,nj->ni", H, torch.distributions.dirichlet.Dirichlet(diri).mean)
    norm = torch.einsum("ij,ij->", mean, mean)
    tvar = torch.einsum("ij,nj,ij->", H, torch.distributions.dirichlet.Dirichlet(diri).variance, H)
    entropy = torch.distributions.dirichlet.Dirichlet(diri).entropy().sum()
    loss = ((norm + tvar) / (2 * sigma) - entropy) / Y.shape[0]
    return loss


def VIA_GD(Y, H, sigma=1e-2, startRoundsd=20, roundsTotal=200):
    n = Y.shape[0]
    m, d = H.shape
    diri = torch.ones((n, d), dtype=torch.float32, device=device)
    diri.requires_grad_(False)
    H.requires_grad_(False)
    i = 0
    while i < startRoundsd:
        i += 1

        diri.requires_grad_(False)
        diri = gradient_descent_with_backtracking_step(lambda dir: min_func(Y, H, dir, sigma), diri)

    i = 0
    while i < roundsTotal:
        i += 1

        eta = diri.sum(dim=-1)
        left = torch.einsum("ij,ik,i->jk", Y, diri, 1 / eta)
        right = (torch.einsum("ij,ni,n->ij", torch.eye(d, dtype=torch.float32, device=device), diri,
                              1 / ((1 + eta) * eta)) + torch.einsum(
            "ij,ik,i->jk",
            diri, diri,
            1 / ((
                         1 + eta) * eta)))
        H = left @ torch.linalg.inv(right)
        diri = gradient_descent_with_backtracking_step(lambda dir: min_func(Y, H, dir, sigma), diri)

    return H.cpu().numpy()
