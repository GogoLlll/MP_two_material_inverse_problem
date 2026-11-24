import numpy as np

class MLP:
    def __init__(self, sizes=(6, 64, 64, 2), seed=11, init="lecun"):
        self.sizes = tuple(sizes)
        rng = np.random.default_rng(seed)
        self.W, self.b = [], []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            var = 1.0 / n_in if init == "lecun" else 2.0 / (n_in + n_out)
            self.W.append(rng.standard_normal((n_out, n_in)) * np.sqrt(var))
            self.b.append(np.zeros(n_out))

    @property
    def params(self):
        out = []
        for W, b in zip(self.W, self.b):
            out += [W, b]
        return out

    def param_names(self):
        names = []
        for l in range(1, len(self.W) + 1):
            names += [f"W{l}", f"b{l}"]
        return names

    def n_params(self):
        return sum(p.size for p in self.params)

    def copy_params(self):
        return [p.copy() for p in self.params]

    def set_params(self, values):
        for p, v in zip(self.params, values):
            p[...] = v

    def save(self, path, **extra):
        arrays = dict(zip(self.param_names(), self.params))
        np.savez(path, sizes=np.array(self.sizes), **arrays, **extra)

    @classmethod
    def load(cls, path):
        f = np.load(path)
        net = cls(tuple(int(s) for s in f["sizes"]), seed=0)
        net.set_params([f[name] for name in net.param_names()])
        return net

    def forward(self, Z):
        A = Z
        cache = [A]
        L = len(self.W)
        for l, (W, b) in enumerate(zip(self.W, self.b)):
            H = A @ W.T + b
            if l < L - 1:
                A = np.tanh(H)
                cache.append(A)
            else:
                A = H
        return A, cache

    def predict(self, Z):
        return self.forward(Z)[0]

    @staticmethod
    def loss(out, target):
        B = out.shape[0]
        return 0.5 * np.sum((out - target) ** 2) / B

    def loss_and_grads(self, Z, target):
        out, cache = self.forward(Z)
        B = Z.shape[0]
        L = len(self.W)
        G = (out - target) / B
        gW, gb = [None] * L, [None] * L
        for l in range(L - 1, -1, -1):
            A_prev = cache[l]
            gW[l] = G.T @ A_prev
            gb[l] = G.sum(axis=0)
            if l > 0:
                G = (G @ self.W[l]) * (1.0 - A_prev**2)
        grads = []
        for gw, gbb in zip(gW, gb):
            grads += [gw, gbb]
        return self.loss(out, target), grads

class SGD:
    def __init__(self, params, lr):
        self.params, self.lr = params, lr

    def step(self, grads):
        for p, g in zip(self.params, grads):
            p -= self.lr * g

class Momentum:
    def __init__(self, params, lr, beta=0.9):
        self.params, self.lr, self.beta = params, lr, beta
        self.buf = [np.zeros_like(p) for p in params]

    def step(self, grads):
        for p, g, v in zip(self.params, grads, self.buf):
            v *= self.beta
            v += g
            p -= self.lr * v

class Adam:
    def __init__(self, params, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.lr = params, lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.s = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        self.t += 1
        c1 = 1.0 - self.b1**self.t
        c2 = 1.0 - self.b2**self.t
        for p, g, m, s in zip(self.params, grads, self.m, self.s):
            m *= self.b1
            m += (1.0 - self.b1) * g
            s *= self.b2
            s += (1.0 - self.b2) * g * g
            p -= self.lr * (m / c1) / (np.sqrt(s / c2) + self.eps)

OPTIMIZERS = {"sgd": SGD, "momentum": Momentum, "adam": Adam}
