import numpy as np

class LightCurveSimulator:
    def __init__(self, tau=100.0, sigma=0.2, mean_flux=10.0):
        self.tau = tau
        self.sigma = sigma
        self.mean_flux = mean_flux

    def generate_drw_source(self, t_obs):
        flux = np.zeros_like(t_obs)
        flux[0] = np.random.normal(0, self.sigma)
        for i in range(1, len(t_obs)):
            dt = t_obs[i] - t_obs[i-1]
            decay = np.exp(-dt / self.tau)
            variance = (self.sigma**2) * (1.0 - decay**2)
            flux[i] = decay * flux[i-1] + np.random.normal(0, np.sqrt(variance))
        return flux + self.mean_flux

    def generate_unresolved_curve(self, t_obs, source_flux, t_A, t_B, mu_A, mu_B, noise_level=0.02):
        flux_A = mu_A * np.interp(t_obs - t_A, t_obs, source_flux)
        flux_B = mu_B * np.interp(t_obs - t_B, t_obs, source_flux)
        joint_flux = flux_A + flux_B
        noise = np.random.normal(0, noise_level * np.mean(joint_flux), size=len(t_obs))
        obs_flux = joint_flux + noise
        return obs_flux, flux_A, flux_B
