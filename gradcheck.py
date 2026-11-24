import numpy as np

def central_difference(net, Z, target, p_index, flat_index, eps=1e-5):
    p = net.params[p_index]
    idx = np.unravel_index(flat_index, p.shape)
    old = p[idx]
    p[idx] = old + eps
    lp = net.loss(net.predict(Z), target)
    p[idx] = old - eps
    lm = net.loss(net.predict(Z), target)
    p[idx] = old
    return (lp - lm) / (2 * eps)

def check(net, Z, target, picks, eps=1e-5):
    _, grads = net.loss_and_grads(Z, target)
    names = net.param_names()
    rows, g, b = [], [], []
    for pi, fi in picks:
        gi = grads[pi].ravel()[fi]
        bi = central_difference(net, Z, target, pi, fi, eps)
        idx = tuple(int(i) for i in np.unravel_index(fi, net.params[pi].shape))
        rows.append((f"{names[pi]}{list(idx)}", gi, bi))
        g.append(gi)
        b.append(bi)
    g, b = np.array(g), np.array(b)
    rel = np.linalg.norm(g - b) / max(1e-8, np.linalg.norm(g) + np.linalg.norm(b))
    return rows, rel

def pick_parameters(net, rng, n_weights_per_layer=4, bias_counts=(1, 1, 2)):
    picks = []
    n_layers = len(net.W)
    for l in range(n_layers):
        pi = 2 * l
        for fi in rng.choice(net.W[l].size, n_weights_per_layer, replace=False):
            picks.append((pi, int(fi)))
    for l, cnt in zip(range(n_layers), bias_counts):
        pi = 2 * l + 1
        for fi in rng.choice(net.b[l].size, cnt, replace=False):
            picks.append((pi, int(fi)))
    return picks
