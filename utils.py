import torch


class System:
    def __init__(
        self,
        X_0,
        f,
        h,
        N_paths,
        sys_noise,
        sys_noise_probs,
        obs_noise,
        obs_noise_probs,
    ):
        self.x = X_0[:, torch.newaxis].repeat(1, N_paths)

        self.f = f
        self.h = h
        self.N_paths = N_paths
        self.sys_noise = sys_noise
        self.sys_noise_probs = sys_noise_probs
        self.obs_noise = obs_noise
        self.obs_noise_probs = obs_noise_probs

    def step(self):
        self.x = self.f(self.x)
        for i in range(len(self.sys_noise)):
            self.x[i] += self.sys_noise[i][
                torch.multinomial(self.sys_noise_probs[i], self.N_paths, True)
            ]
        return self.x

    def get_obs(self):
        obs = self.h(self.x)
        for i in range(len(self.obs_noise)):
            obs[i] += self.obs_noise[i][
                torch.multinomial(self.obs_noise_probs[i], self.N_paths, True)
            ]
        return obs


class ExtendedKF2:
    def __init__(
        self,
        f,
        h,
        grad_f,
        grad_h,
        hessian_f,
        hessian_h,
        X_init,
        cov_init,
        cov_sys,
        cov_obs,
        N_paths,
    ):
        self.f = f
        self.h = h
        self.grad_f = grad_f
        self.grad_h = grad_h
        self.hessian_f = hessian_f
        self.hessian_h = hessian_h

        self.N_paths = N_paths

        self.cov_sys = cov_sys[:, :, torch.newaxis].expand(-1, -1, N_paths)
        self.cov_obs = cov_obs[:, :, torch.newaxis].expand(-1, -1, N_paths)

        self.x = X_init[:, torch.newaxis].repeat(1, N_paths)
        self.k = cov_init[:, :, torch.newaxis].repeat(1, 1, N_paths)

    def predict(self):
        grad = self.grad_f(self.x)
        hessian = self.hessian_f(self.x)

        self.x = self.f(self.x) + torch.einsum("jki,ipkj->pi", self.k, hessian) / 2
        self.k = (
            torch.einsum("ijk,kli,ipl->jpi", grad, self.k, grad)
            + torch.einsum("liab,bcl,ljcd,dal->ijl", hessian, self.k, hessian, self.k)
            / 2
            + self.cov_sys
        )

        return self.x

    def update(self, y):
        grad = self.grad_h(self.x)
        hessian = self.hessian_h(self.x)

        y_pred = self.h(self.x) + torch.einsum("jki,ipkj->pi", self.k, hessian) / 2

        mu = torch.einsum("ijl,lkj->ikl", self.k, grad)
        q = torch.einsum("lij,jkl->ikl", grad, mu) + self.cov_obs
        q_inv = torch.empty_like(q)
        for i in range(self.N_paths):
            q_inv[:, :, i] = torch.linalg.inv(q[:, :, i])

        kalman_gain = torch.einsum("ijl,jkl->ikl", mu, q_inv)
        self.x += torch.einsum("ijl,jl->il", kalman_gain, y - y_pred)
        self.k -= torch.einsum("abl,cbl->acl", kalman_gain, mu)

        return self.x


class BootstrapPF:
    def __init__(
        self,
        f,
        h,
        X_init,
        N_samples,
        N_paths,
        sys_noise,
        sys_noise_probs,
        sys_std,
        obs_noise,
        obs_noise_probs,
        obs_std,
    ):
        self.f = f
        self.h = h
        self.N_samples = N_samples
        self.N_paths = N_paths

        self.sys_noise = sys_noise
        self.sys_noise_probs = sys_noise_probs
        self.sys_std = sys_std

        self.obs_noise = obs_noise
        self.obs_noise_probs = obs_noise_probs
        self.obs_std = obs_std

        self.X = X_init[:, torch.newaxis].repeat(1, self.N_samples * self.N_paths)
        for i in range(2):
            self.X[i] += self.sys_noise[i][
                torch.multinomial(
                    self.sys_noise_probs[i], self.N_samples * self.N_paths, True
                )
            ]
        self.X += self.sys_std[:, torch.newaxis] * torch.randn_like(self.X)
        self.w = torch.ones(self.N_samples, self.N_paths) / self.N_samples

    def norm_cdf(self, x, mu, sigma):
        return torch.distributions.Normal(mu, sigma).log_prob(x).exp()

    def cond_density(self, y):
        y_pred = self.h(self.X).view(-1, self.N_samples, self.N_paths)
        result = torch.ones(self.N_samples, self.N_paths)
        for i in range(len(self.obs_noise)):
            tmp = 0
            for prob, noise in zip(self.obs_noise_probs[i], self.obs_noise[i]):
                tmp += prob * self.norm_cdf(y[i], y_pred[i] + noise, self.obs_std[i])
            result *= tmp
        return result

    def resample(self):
        self.X = self.X.view(-1, self.N_samples, self.N_paths)

        indeces, *_ = torch.where(10 / (self.w**2).sum(dim=0) < self.N_samples)
        for i in indeces:
            self.X[:, :, i] = self.X[
                :,
                torch.multinomial(self.w[:, i], self.N_samples, True),
                i,
            ]
            self.w[:, i] = torch.ones(self.N_samples) / self.N_samples

        self.X = self.X.view(-1, self.N_samples * self.N_paths)

    def predict(self):
        self.X = self.f(self.X)
        for i in range(2):
            self.X[i] += self.sys_noise[i][
                torch.multinomial(
                    self.sys_noise_probs[i], self.N_samples * self.N_paths, True
                )
            ]
        self.X += self.sys_std[:, torch.newaxis] * torch.randn_like(self.X)

        self.X = self.X.view(-1, self.N_samples, self.N_paths)

        m = torch.sum(self.X * self.w, dim=1)
        centered_X = self.X - m[:, torch.newaxis]

        self.X = self.X.view(-1, self.N_samples * self.N_paths)
        return m, torch.sqrt(torch.sum(centered_X**2 * self.w, dim=1))

    def update(self, y):
        self.w *= self.cond_density(y)
        self.w += 1e-10
        self.w /= self.w.sum(dim=0)

        self.resample()

        self.X = self.X.view(-1, self.N_samples, self.N_paths)

        m = torch.sum(self.X * self.w, dim=1)
        centered_X = self.X - m[:, torch.newaxis]

        self.X = self.X.view(-1, self.N_samples * self.N_paths)
        return m, torch.sqrt(torch.sum(centered_X**2 * self.w, dim=1))
