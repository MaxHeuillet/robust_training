import numpy as np
from scipy.optimize import curve_fit

class FitExpDecay_v2:

    def __init__(self, c_fixed):
        self.c_fixed = c_fixed
    
    @staticmethod
    def exponential_decay_fixed_a(t, b, c, a):
        return a * np.exp(-b * t) + c
    
    @staticmethod
    def exponential_decay_fixed_a_c_zero(t, b, a):
        return a * np.exp(-b * t)
    
    def fit(self, loss_values):
        t = np.arange(len(loss_values))
        a = loss_values[0]  # Fix a as the first loss value
        n = len(loss_values)
        
        if self.c_fixed:
            # c is fixed to zero
            # Use curve_fit to fit only b
            popt, _ = curve_fit(
                lambda t, b: FitExpDecay.exponential_decay_fixed_a_c_zero(t, b, a),
                t,
                loss_values,
                p0=(0.1,),             # Initial guess for b
                bounds=([0], [np.inf])  # Bounds: b >= 0
            )
            b = popt[0]
            c = 0  # Since c is fixed to zero
            return (a, b, c, n)
        else:
            # c is not fixed
            # Use curve_fit with bounds to ensure c >= 0
            popt, _ = curve_fit(
                lambda t, b, c: FitExpDecay.exponential_decay_fixed_a(t, b, c, a),
                t,
                loss_values,
                p0=(0.1, 0),            # Initial guesses for b and c
                bounds=([0, 0], [np.inf, np.inf])  # Bounds: b >= 0, c >= 0
            )
            b, c = popt
            return (a, b, c, n)  # Return a, b, c
        
    def predict(self, fit, last_observed_value):
        a, b, c, n = fit
        # Predict the next observation value (n+1)
        next_value = self.exponential_decay_fixed_a(n, b, c, a)
        # Calculate the decay rate using the actual last observed value
        decay_rate = (last_observed_value - next_value) / last_observed_value
        return decay_rate
    
    def fit_predict(self, loss_values):
        fit = self.fit(loss_values)
        last_observed_value = loss_values[-1]  # Use the actual last observed value
        pred = self.predict(fit, last_observed_value)
        a, b, c, n = fit
        return (a, b, c, n, pred)
