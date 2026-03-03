# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:57:36 2024

@author: Arp00003
"""
from abc import abstractmethod

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit, minimize

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


def mask_valid(*data: tuple[np.array]):
    """Returns a mask which observations (rows) contain valid values."""
    matrix = np.concatenate(data, axis=1)
    is_valid = np.isfinite(matrix).all(axis=1)
    return is_valid


def remove_nan(*data: tuple[np.array]):
    """Remove observations (rows) with NaN values."""
    is_valid = mask_valid(*data)
    cleaned_data = [datum[is_valid] for datum in data]
    return cleaned_data


def prepare_regression_data(y: pd.DataFrame = None, *X: tuple[pd.DataFrame]):
    """
    Flattens data, removes NaNs, and concatenates regressors
    """
    # Flatten data
    y = y.to_numpy().flatten().reshape((-1, 1))
    # find max, and make sure it's float, so skip all strings
    # print('max X', X_max)
    X = [regressor.to_numpy().flatten().reshape((-1, 1)) for regressor in X]
    X_max = max(max(X))
    # Remove NaNs
    y, *X = remove_nan(y, *X)
    # Concatenate regressors to matrix
    X = np.concatenate(X, axis=1)
    
    return y, X, X_max


class OLS_Model:
    def __init__(self, y: pd.DataFrame, *X: tuple[pd.DataFrame]):
        # Prepare regression data
        y, X, self.X_max = prepare_regression_data(y, *X)
        self._y = self._transform_y(y)
        self._X = self._transform_X(X)
        y, self._y, self._X = remove_nan(y, self._y, self._X)
        # Create LinearRegression object
        self._lin_reg = LinearRegression()
        # Estimate coefficients
        self._lin_reg.fit(self._X, self._y)
        # Estimate R^2
        self._r2 = self._lin_reg.score(self._X, self._y)
        # Estimate RMSE
        y_pred = self.predict_transformed(self._X)  # y in transformed form
        y_pred = self._inverse_transform_y(y_pred)  # transform to original unit
        self._rmse = mean_squared_error(y, y_pred) # squared=False
        # Store coefficients
        self._coefs = [self._lin_reg.intercept_[0], *self._lin_reg.coef_[0]]
        
    @abstractmethod
    def _transform_X(self, X: np.array):
        pass
    
    @abstractmethod
    def _transform_y(self, y: np.array):
        pass
    
    @abstractmethod
    def _inverse_transform_y(self, y: np.array):
        pass
    
    @property
    def coefs(self):
        return self._coefs
    
    @property
    def r2(self):
        return self._r2
    
    @property
    def rmse(self):
        return self._rmse
    
    def predict(self, X: np.array):
        """
        Predict outcome of original data.
        """
        # print('X predict shape', X.shape)
        X = self._transform_X(X)  # transform X
        is_valid = mask_valid(X)
        # print(sum(is_valid), 'valid observations')
        y_pred = np.empty((X.shape[0], 1))
        y_pred[~is_valid] = np.nan  # assign NaN to invalid X data
        y_pred[is_valid] = self.predict_transformed(X[is_valid])  # predict y
        y_pred[is_valid] = self._inverse_transform_y(y_pred[is_valid])  # inverse-transform y
        return y_pred
    
    def predict_transformed(self, X: np.array):
        """
        Predict outcome with transformed data, and therefore returns predicted 
        y in transformed form.
        """
        y_pred = self._lin_reg.predict(X)
        return y_pred

#%% NLS class

class NLS_Model:
    def __init__(self, y: pd.DataFrame, *X: tuple[pd.DataFrame], **kwargs):
        bounds = kwargs.get("bounds")
        # Prepare regression data
        y, X, self.X_max = prepare_regression_data(y, *X)
        self._y = self._transform_y(y)
        self._X = self._transform_X(X)
        # remove nan again in case of division by zero error, log(-x), ... after transformation
        y, self._y, self._X = remove_nan(y, self._y, self._X)
        # Fit parameters for NonLinearRegression
        self._y = self._y.reshape(-1)
        self._X = self._X.reshape(-1)  # TODO: Implement for multiple regressors

        self._coefs, _ = curve_fit(self._model_func, 
                                   self._X, 
                                   self._y, maxfev=10_000, 
                                   bounds=bounds) 
        self._r2 = np.nan
        # Estimate RMSE
        y_pred = self.predict_transformed(self._X)  # y in transformed form
        y_pred = self._inverse_transform_y(y_pred)  # transform to original unit
        self._rmse = mean_squared_error(y, y_pred) #squared=False
        
    @abstractmethod
    def _transform_X(self, X: np.array):
        pass
    
    @abstractmethod
    def _transform_y(self, y: np.array):
        pass
    
    @abstractmethod
    def _inverse_transform_y(self, y: np.array):
        pass
    
    @abstractmethod
    def _model_func(self, X, *args):
        pass
    
    @property
    def coefs(self):
        return self._coefs
    
    @property
    def r2(self):
        return self._r2
    
    @property
    def rmse(self):
        return self._rmse
    
    def predict(self, X: np.array):
        """
        Predict outcome of original data.
        """
        # print('X predict shape', X.shape)
        X = self._transform_X(X)  # transform X
        is_valid = mask_valid(X)
        # print(sum(is_valid), 'valid observations')
        y_pred = np.empty((X.shape[0], 1))
        y_pred[~is_valid] = np.nan  # assign NaN to invalid X data
        y_pred[is_valid] = self.predict_transformed(X[is_valid])  # predict y
        y_pred[is_valid] = self._inverse_transform_y(y_pred[is_valid])  # inverse-transform y
        return y_pred
    
    def predict_transformed(self, X: np.array):
        """
        Predict outcome with transformed data, and therefore returns predicted 
        y in transformed form.
        """
        y_pred = self._model_func(X, *self._coefs)
        return y_pred

#%%

class Log_Log_Model(OLS_Model):
    def _transform_X(self, X: np.array):
        X = np.asarray(X, dtype=float)
        X[X <= 0] = np.nan
        return np.log(X)
    
    def _transform_y(self, y: np.array):
        y = np.asarray(y, dtype=float)
        y[y <= 0] = np.nan
        return np.log(y)
    
    def _inverse_transform_y(self, y: np.array):
        return np.exp(y)


class Semi_Log_Model(OLS_Model):
    def _transform_X(self, X: np.array):
        X = np.asarray(X, dtype=float)
        X[X <= 0] = np.nan
        return np.log(X)
    
    def _transform_y(self, y: np.array):
        return y
    
    def _inverse_transform_y(self, y: np.array):
        return y


class Log_Inverse_Model(OLS_Model):
    def _transform_X(self, X: np.array):
        X = np.asarray(X, dtype=float)
        X[X == 0] = np.nan
        return np.divide(1, X)
    
    def _transform_y(self, y: np.array):
        y = np.asarray(y, dtype=float)
        y[y <= 0] = np.nan
        return np.log(y)
    
    def _inverse_transform_y(self, y: np.array):
        return np.exp(y)


class Log_Log_Inverse_Model(OLS_Model):
    def _transform_X(self, X: np.array):
        X = np.asarray(X, dtype=float)
        X[X <= 0] = np.nan
        return np.concatenate([np.log(X), 1 / X], axis=1)
    
    def _transform_y(self, y: np.array):
        y = np.asarray(y, dtype=float)
        y[y <= 0] = np.nan
        return np.log(y)
    
    def _inverse_transform_y(self, y: np.array):
        return np.exp(y)


class Log_Log_Square_Model(OLS_Model):
    def _transform_X(self, X: np.array):
        X = np.asarray(X, dtype=float)
        X[X <= 0] = np.nan
        return np.concatenate([np.log(X), np.power(np.log(X), 2)], axis=1)
    
    def _transform_y(self, y: np.array):
        y = np.asarray(y, dtype=float)
        y[y <= 0] = np.nan
        return np.log(y)
    
    def _inverse_transform_y(self, y: np.array):
        return np.exp(y)

#%% Define non-linear inverse model (NLI): C = a × e(B/GDP)

class NLI_Model(NLS_Model):
    def __init__(self, y: pd.DataFrame, *X: tuple[pd.DataFrame], **kwargs):
        bounds = kwargs.get("bounds")
        # check if bound is none or if its has the correct length
        if bounds is None or (len(bounds) != 2 or len(bounds[0]) != 3 or len(bounds[1]) != 3):
            bounds = ([-300, -300, -300], [300, 300, 300])

        super().__init__(y, *X, bounds=bounds)

    def _transform_X(self, X: np.array):
        return X
    
    def _transform_y(self, y: np.array):
        return y
    
    def _inverse_transform_y(self, y: np.array):
        return y
        
    def _model_func(self, X: np.ndarray, a: float, b: float, c: float):
        ''' function of non linear inverse (nli)'''
        return a * np.exp(b / X) +c*0


class GOMPERTZ_Model(NLS_Model):
    def __init__(self, y: pd.DataFrame, *X: tuple[pd.DataFrame], **kwargs):
        bounds = kwargs.get("bounds")
        # check if bound is none or if its has the correct length
        if bounds is None or (len(bounds) != 2 or len(bounds[0]) != 3 or len(bounds[1]) != 3):
            bounds = ([0, 0, 0], [10, 10, 20])

        super().__init__(y, *X, bounds=bounds)

    def _transform_X(self, X: np.array):
        # Normalize X by its maximum value so that SciPy.curve_fit has a stable approximation.
        # If X is too large, then the Gompertz function evaluates to the asymptote 'a'.
        # TODO: Normalize also for other NLS_models below.
        X = np.divide(X, self.X_max) 
        return X
    
    def _transform_y(self, y: np.array):
        return y
    
    def _inverse_transform_y(self, y: np.array):
        return y
        
    def _model_func(self, X: np.ndarray, a: float, b: float, c: float):
        ''' gompertz function '''
        return a * np.exp(-b * np.exp(-c * X))


class LG_Model(NLS_Model): #logistic_growth_model
    def __init__(self, y: pd.DataFrame, *X: tuple[pd.DataFrame], **kwargs):
        bounds = kwargs.get("bounds")
        # check if bound is none or if its has the correct length
        if bounds is None or (len(bounds) != 2 or len(bounds[0]) != 3 or len(bounds[1]) != 3):
            bounds = ([0, 0, 0], [300, 300, 300])

        super().__init__(y, *X, bounds=bounds)    
    
    def _transform_X(self, X: np.array):
        X = np.divide(X, self.X_max)
        return X
    
    def _transform_y(self, y: np.array):
        return y
    
    def _inverse_transform_y(self, y: np.array):
        return y
        
    def _model_func(self, X: np.ndarray, a: float, b: float, c: float):
        ''' lg function '''
        return c / (1+ a * np.exp(-b * X))
    

class BW_Model(NLS_Model):
    def __init__(self, y: pd.DataFrame, *X: tuple[pd.DataFrame], **kwargs):
        bounds = kwargs.get("bounds")
        # check if bound is none or if its has the correct length
        if bounds is None or (len(bounds) != 2 or len(bounds[0]) != 3 or len(bounds[1]) != 3):
            bounds = ([0, 0, 0], [300, 300, 300])

        super().__init__(y, *X, bounds=bounds)

    def _transform_X(self, X: np.array):
        X = np.divide(X, self.X_max)
        return X
    
    def _transform_y(self, y: np.array):
        return y
    
    def _inverse_transform_y(self, y: np.array):
        return y
        
    def _model_func(self, X: np.ndarray, a: float, b: float, c: float):
        ''' bw function '''
        return a - (a - b)*np.exp(-c*X)


class Log_Gauss_Saturate_Model(NLS_Model):
    def __init__(self, y: pd.DataFrame, *X: tuple[pd.DataFrame], **kwargs):
        bounds = kwargs.get("bounds")
        # check if bound is none or if its has the correct length
        if bounds is None or (len(bounds) != 2 or len(bounds[0]) != 4 or len(bounds[1]) != 4):
            # raise ValueError("Bounds must be a tuple of two lists with 4 elements each.")
            # Default: 4 parameters
            bounds = ([0, 0, 0, 0], [300, 300, 300, 300])

        super().__init__(y, *X, bounds=bounds)

    def _transform_X(self, X: np.array):
        X = np.asarray(X, dtype=float)
        X = np.divide(X, self.X_max)
        X[X <= 0] = np.nan
        return X

    def _transform_y(self, y: np.array):
        return y

    def _inverse_transform_y(self, y: np.array):
        return y

    def _model_func(self, X, a, b, c, d):
        logX = np.log(X)
        return a * np.exp(-((logX - b) ** 2) / (2 * c ** 2)) + d