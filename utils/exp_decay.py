import numpy as np
from scipy.optimize import curve_fit

class FitExpDecay:

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
        
        # if n < 3:  # Minimum data length check
        #     return None  # Not enough data to fit
        
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
        
    def predict(self, fit):
        a, b, c, n = fit
        if self.c_fixed:
            return FitExpDecay.exponential_decay_fixed_a(n, b, c, a)
        else:
            return FitExpDecay.exponential_decay_fixed_a_c_zero(n, b, a)
        
    def fit_predict(self,loss_values):
        fit = self.fit(loss_values)
        # print(fit)
        pred = self.predict(fit)
        # print(pred)
        a, b, c, n = fit
        return (a, b, c, n, pred)