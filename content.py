# --------------------------------------------------------------------------
# ----------------  System Analysis and Decision Making --------------------
# --------------------------------------------------------------------------
#  Assignment 1: Linear regression
#  Authors: A. Gonczarek, J. Kaczmar, S. Zareba
#  2017
# --------------------------------------------------------------------------

import numpy as np
from utils import polynomial

def mean_squared_error(x, y, w):
    '''
    :param x: input vector Nx1
    :param y: output vector Nx1
    :param w: model parameters (M+1)x1
    :return: mean squared error between output y
    and model prediction for input x and parameters w
    '''

    sum = 0
    N = len(x)
    for i in range(0,N):
        pred_yi = polynomial(x[i],w)
        sum += (y[i] - pred_yi)**2

    return sum / N


def design_matrix(x_train,M):
    '''
    :param x_train: input vector Nx1
    :param M: polynomial degree 0,1,2,...
    :return: Design Matrix Nx(M+1) for M degree polynomial
    '''

    N = len(x_train)
    dm = np.ndarray(shape=(N, M+1), dtype=float, order='F')

    for m in range(0,M+1):
       for n in range(0,N):
            dm[n][m] = x_train[n]**m

    return dm


def least_squares(x_train, y_train, M):
    '''
    :param x_train: training input vector  Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial
    '''
    dm = design_matrix(x_train,M)
    w = np.dot(dm.T,dm)
    w = np.linalg.inv(w)
    w = np.dot(w,dm.T)
    w = np.dot(w,y_train)
    return w, mean_squared_error(x_train,y_train,w)[0]

def regularized_least_squares(x_train, y_train, M, regularization_lambda):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param M: polynomial degree
    :param regularization_lambda: regularization parameter
    :return: tuple (w,err), where w are model parameters and err mean squared error of fitted polynomial with l2 regularization
    '''
    dm = design_matrix(x_train, M)
    w = np.dot(dm.T, dm)
    w = w + regularization_lambda * np.eye(M+1)
    w = np.linalg.inv(w)
    w = np.dot(w, dm.T)
    w = np.dot(w, y_train)
    return w, mean_squared_error(x_train, y_train, w)[0]

def model_selection(x_train, y_train, x_val, y_val, M_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M_values: array of polynomial degrees that are going to be tested in model selection procedure
    :return: tuple (w,train_err, val_err) representing model with the lowest validation error
    w: model parameters, train_err, val_err: training and validation mean squared error
    '''

    w_best, x_train_best = least_squares(x_train,y_train,M_values[0])
    y_val_best = mean_squared_error(x_val,y_val,w_best)[0]

    for m in M_values[1:]:
        w_temp, x_train_temp = least_squares(x_train,y_train,m)
        y_val_temp = mean_squared_error(x_val,y_val,w_temp)[0]
        if y_val_temp < y_val_best:
            w_best = w_temp
            x_train_best = x_train_temp
            y_val_best = y_val_temp

    return w_best, x_train_best, y_val_best


def regularized_model_selection(x_train, y_train, x_val, y_val, M, lambda_values):
    '''
    :param x_train: training input vector Nx1
    :param y_train: training output vector Nx1
    :param x_val: validation input vector Nx1
    :param y_val: validation output vector Nx1
    :param M: polynomial degree
    :param lambda_values: array of regularization coefficients are going to be tested in model selection procedurei
    :return:  tuple (w,train_err, val_err, regularization_lambda) representing model with the lowest validation error
    (w: model parameters, train_err, val_err: training and validation mean squared error, regularization_lambda: the best value of regularization coefficient)
    '''

    w_best, x_train_best = regularized_least_squares(x_train, y_train, M, lambda_values[0])
    y_val_best = mean_squared_error(x_val, y_val, w_best)[0]
    l_best = lambda_values[0]

    for l in lambda_values[1:]:
        w_temp, x_train_temp = regularized_least_squares(x_train, y_train, M, l)
        y_val_temp = mean_squared_error(x_val, y_val, w_temp)[0]
        if y_val_temp < y_val_best:
            w_best = w_temp
            x_train_best = x_train_temp
            y_val_best = y_val_temp
            l_best = l

    return w_best, x_train_best, y_val_best, l_best