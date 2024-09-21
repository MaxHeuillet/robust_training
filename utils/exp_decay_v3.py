import numpy as np
from scipy.optimize import lsq_linear

class FitExpDecay_v3:

    def __init__(self, ):
        pass

    
    def linearize_exponential_decay(self, t, loss_values):
        """
        Linearize the exponential decay by taking the logarithm of loss values.
        y = a * exp(-b * t) becomes log(y) = -b * t + log(a)
        """
        # Take logarithm of loss values to linearize the problem
        log_loss_values = np.log(loss_values)
        # Design matrix with intercept term
        X = np.vstack([-t, np.ones_like(t)]).T  # Design matrix [-t, 1]
        return log_loss_values, X

    def generate_adaptive_prior(self, loss_values):
        t = np.arange(len(loss_values)) if len(loss_values)>=2 else [0,1]
        b_prior = np.log(loss_values[0] / loss_values[-1]) / (t[-1] - t[0])  # Rough estimate of the decay rate
        c_prior = np.min(loss_values)  # Assume the smallest loss is close to c
        return [b_prior, c_prior]
    
    def fit(self, loss_values, prior):
        n = len(loss_values)
        t = np.arange(n)
        
        # Linearize the problem: log(y) = -b * t + log(a)
        log_loss_values, X = self.linearize_exponential_decay(t, loss_values)
        print(log_loss_values.shape, X.shape)

        # Solve for b and log(a)
        # For exponential decay, b >= 0
        bounds = ([0, -np.inf], [np.inf, np.inf])  # b >= 0

        res = lsq_linear(X, log_loss_values, bounds=bounds)
        b = res.x[0]
        c = 0
        log_a = res.x[1]
        a_est = np.exp(log_a)
        return (a_est, b, c, n)
    
    def predict(self, fit):
        a, b, c, n = fit
        # Predict the next observation value at time n
        last_observed_value = a * np.exp(-b * (n - 1)) + c
        next_value = a * np.exp(-b * n) + c
        # Calculate the decay rate using the actual last observed value
        decay_rate = (last_observed_value - next_value) / last_observed_value
        return decay_rate

    def fit_predict(self, loss_values, beta=0.1, ceta=0.1):

        self.total += 1
        prior = self.generate_adaptive_prior(loss_values)

        try:
            fit = self.fit(loss_values, prior)
        except:
            fit = (loss_values[-1], prior[0], prior[1], len(loss_values) )
            self.fail +=1
        pred = self.predict(fit)
        a, b, c, n = fit
        return (a, b, c, n, pred)
        
    def predict(self, fit):
        a, b, c, n = fit
        # Predict the next observation value (n+1)
        last_observed_value = a * np.exp(-b * (n-1)) + c
        next_value = a * np.exp(-b * n) + c
        # Calculate the decay rate using the actual last observed value
        decay_rate = (last_observed_value - next_value) / last_observed_value
        return decay_rate

    def reset_counters(self,):
        self.fail = 0
        self.total = 0