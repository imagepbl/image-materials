# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 18:57:36 2024

@author: Arp00003
"""
from abc import abstractmethod
import inspect

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from scipy.optimize import curve_fit, minimize
from scipy import stats

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
    if X and all(hasattr(frame, 'index') and hasattr(frame, 'columns') for frame in (y, *X)):
        common_index = y.index
        common_columns = y.columns

        for regressor in X:
            common_index = common_index.intersection(regressor.index)
            common_columns = common_columns.intersection(regressor.columns)

        if len(common_index) == 0 or len(common_columns) == 0:
            raise ValueError('Regression inputs do not share a common index/column set.')

        y = y.loc[common_index, common_columns]
        X = [regressor.loc[common_index, common_columns] for regressor in X]

    # Flatten data
    y = y.to_numpy().flatten().reshape((-1, 1))
    # find max, and make sure it's float, so skip all strings
    # print('max X', X_max)
    X = [regressor.to_numpy().flatten().reshape((-1, 1)) for regressor in X]
    X_max = max(float(np.nanmax(regressor)) for regressor in X)
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
        self._build_statistics()

    def _build_statistics(self):
        """Build parameter-level and model-level statistics for OLS."""
        n_obs = self._X.shape[0]
        n_features = self._X.shape[1]
        n_params = n_features + 1
        dof = n_obs - n_params

        if dof <= 0:
            self._std_errors = [np.nan] * n_params
            self._p_values = [np.nan] * n_params
            self._confidence_intervals = [(np.nan, np.nan)] * n_params
            self._p_value = np.nan
            param_names = ["intercept", *[f"x{i + 1}" for i in range(n_features)]]
            self._stats_summary = pd.DataFrame(
                {
                    "coef": self._coefs,
                    "std_error": self._std_errors,
                    "p_value": self._p_values,
                    "ci_95_lower": [ci[0] for ci in self._confidence_intervals],
                    "ci_95_upper": [ci[1] for ci in self._confidence_intervals],
                },
                index=param_names,
            )
            return

        y_pred = self.predict_transformed(self._X)
        residuals = self._y - y_pred
        ss_res = float(np.sum(np.square(residuals)))
        ss_tot = float(np.sum(np.square(self._y - np.mean(self._y))))

        x_design = np.concatenate([np.ones((n_obs, 1)), self._X], axis=1)
        xtx_inv = np.linalg.pinv(x_design.T @ x_design)
        sigma2 = ss_res / dof
        cov = sigma2 * xtx_inv
        std_errors = np.sqrt(np.diag(cov))

        params = np.asarray(self._coefs, dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            t_stats = np.divide(
                params,
                std_errors,
                out=np.full_like(params, np.nan, dtype=float),
                where=std_errors > 0,
            )

        p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))
        t_crit = stats.t.ppf(0.975, df=dof)
        ci_lower = params - t_crit * std_errors
        ci_upper = params + t_crit * std_errors

        if n_features > 0 and ss_tot > 0 and ss_res >= 0:
            ms_model = (ss_tot - ss_res) / n_features
            ms_error = ss_res / dof
            f_stat = np.nan if ms_error <= 0 else ms_model / ms_error
            self._p_value = 1 - stats.f.cdf(f_stat, n_features, dof) if np.isfinite(f_stat) else np.nan
        else:
            self._p_value = np.nan

        self._std_errors = std_errors.tolist()
        self._p_values = p_values.tolist()
        self._confidence_intervals = list(zip(ci_lower.tolist(), ci_upper.tolist()))
        param_names = ["intercept", *[f"x{i + 1}" for i in range(n_features)]]
        self._stats_summary = pd.DataFrame(
            {
                "coef": self._coefs,
                "std_error": self._std_errors,
                "p_value": self._p_values,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
            },
            index=param_names,
        )
        
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

    @property
    def p_value(self):
        return self._p_value

    @property
    def p_values(self):
        return self._p_values

    @property
    def std_errors(self):
        return self._std_errors

    @property
    def confidence_intervals(self):
        return self._confidence_intervals

    @property
    def stats_summary(self):
        return self._stats_summary.copy()
    
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

        self._coefs, self._pcov = curve_fit(
            self._model_func,
            self._X,
            self._y,
            maxfev=10_000,
            bounds=bounds,
        )
        self._r2 = np.nan
        # Estimate RMSE
        y_pred = self.predict_transformed(self._X)  # y in transformed form
        y_pred = self._inverse_transform_y(y_pred)  # transform to original unit
        self._rmse = mean_squared_error(y, y_pred) #squared=False
        self._build_statistics()

    def _build_statistics(self):
        """Build parameter-level and model-level statistics for NLS."""
        n_obs = self._X.shape[0]
        n_params = len(self._coefs)
        dof = n_obs - n_params

        sig = inspect.signature(self._model_func)
        parameter_names = list(sig.parameters.keys())[1:]
        if len(parameter_names) != n_params:
            parameter_names = [f"param_{i + 1}" for i in range(n_params)]

        y_pred = self.predict_transformed(self._X)
        residuals = self._y - y_pred
        ss_res = float(np.sum(np.square(residuals)))
        ss_tot = float(np.sum(np.square(self._y - np.mean(self._y))))

        if dof <= 0 or self._pcov is None:
            std_errors = np.full(n_params, np.nan, dtype=float)
            p_values = np.full(n_params, np.nan, dtype=float)
            ci_lower = np.full(n_params, np.nan, dtype=float)
            ci_upper = np.full(n_params, np.nan, dtype=float)
            self._p_value = np.nan
        else:
            std_errors = np.sqrt(np.diag(self._pcov))
            params = np.asarray(self._coefs, dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                t_stats = np.divide(
                    params,
                    std_errors,
                    out=np.full_like(params, np.nan, dtype=float),
                    where=std_errors > 0,
                )
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), df=dof))
            t_crit = stats.t.ppf(0.975, df=dof)
            ci_lower = params - t_crit * std_errors
            ci_upper = params + t_crit * std_errors

            model_dof = n_params - 1
            if model_dof > 0 and ss_tot > 0 and ss_res >= 0:
                ms_model = (ss_tot - ss_res) / model_dof
                ms_error = ss_res / dof
                f_stat = np.nan if ms_error <= 0 else ms_model / ms_error
                self._p_value = 1 - stats.f.cdf(f_stat, model_dof, dof) if np.isfinite(f_stat) else np.nan
            else:
                self._p_value = np.nan

        self._std_errors = std_errors.tolist()
        self._p_values = p_values.tolist()
        self._confidence_intervals = list(zip(ci_lower.tolist(), ci_upper.tolist()))
        self._stats_summary = pd.DataFrame(
            {
                "coef": self._coefs,
                "std_error": self._std_errors,
                "p_value": self._p_values,
                "ci_95_lower": ci_lower,
                "ci_95_upper": ci_upper,
            },
            index=parameter_names,
        )
        
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

    @property
    def p_value(self):
        return self._p_value

    @property
    def p_values(self):
        return self._p_values

    @property
    def std_errors(self):
        return self._std_errors

    @property
    def confidence_intervals(self):
        return self._confidence_intervals

    @property
    def stats_summary(self):
        return self._stats_summary.copy()
    
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