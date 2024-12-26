import numpy as np
from scipy.optimize import curve_fit




class FitExpDecay_v2:

    def __init__(self, ):
        pass
    
    @staticmethod
    def exponential_decay(t, b, c, a):
        return a * np.exp(-b * t) + c
        
    def generate_adaptive_prior(self, loss_values):
        t = np.arange(len(loss_values)) if len(loss_values)>=2 else [0,1]
        b_prior = np.log(loss_values[0] / loss_values[-1]) / (t[-1] - t[0])  # Rough estimate of the decay rate
        c_prior = np.min(loss_values)  # Assume the smallest loss is close to c
        return [b_prior, c_prior]
    
    def fit(self, loss_values, prior):

        t = np.arange(len(loss_values))
        a = loss_values[0]  # Fix a as the first loss value
        n = len(loss_values)
        
        # c is not fixed
        # Use curve_fit with bounds to ensure c >= 0
        popt, _ = curve_fit(
                lambda t, b, c: FitExpDecay_v2.exponential_decay(t, b, c, a),
                t,
                loss_values,
                p0=prior,            # Initial guesses for b and c
                bounds=([0, 0], [np.inf, np.inf])  # Bounds: b >= 0, c >= 0
            )
        b, c = popt
        return (a, b, c, n)  # Return a, b, c
        
    def predict(self, fit,):
        a, b, c, n = fit
        # Predict the next observation value (n+1)
        last_observed_value = self.exponential_decay(n-1, b, c, a)
        next_value = self.exponential_decay(n, b, c, a)
        # Calculate the decay rate using the actual last observed value
        decay_rate = (last_observed_value - next_value) / last_observed_value
        return decay_rate
    
    def fit_predict(self, loss_values, ):
        self.total += 1
        prior = self.generate_adaptive_prior(loss_values)
        try:
            fit = self.fit(loss_values, prior)
        except:
            self.fail +=1
            fit = (loss_values[-1], prior[0], prior[1], len(loss_values) )

        # last_observed_value = loss_values[-1]  # Use the actual last observed value
        pred = self.predict(fit,)
        a, b, c, n = fit
        return (a, b, c, n, pred)
    
    def reset_counters(self,):
        self.fail = 0
        self.total = 0
