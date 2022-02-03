import numpy as np
# pip install prettytable
from prettytable import PrettyTable # for visualization purposes
import matplotlib.pyplot as plt
from proj1_helpers import *

DATA_TRAIN_PATH = 'data/train.csv' # TODO: download train data and supply path here
y, tX, ids = load_csv_data(DATA_TRAIN_PATH)

indicator_feature_1 = np.int32(tX[:,0] == -999)
indicator_feature_2 =  np.int32(tX[:,23] == -999) + np.int32(tX[:,24] == -999) + np.int32(tX[:,25] == -999)

def standardize(x):
    """Standardize the original data set."""
    mean_x = np.mean(x, axis=0)
    x = x - mean_x
    std_x = np.std(x, axis=0)
    x = x / std_x
    return x, mean_x, std_x


def de_standardize(x, mean_x, std_x):
    """Reverse the procedure of standardization."""
    x = x * std_x
    x = x + mean_x
    return x

def split_data(x, y, ratio, seed=1):
    """split the dataset based on the split ratio."""
    # set seed
    np.random.seed(seed)
    
    # Randomly choose indexes of train set
    N = x.shape[0]
    
    indexes = np.random.permutation(np.arange(N))
    
    train_indices = indexes[:int(np.round(N*ratio))]
    test_indices = indexes[int(np.round(N*ratio)):]

   
    return x[train_indices,:], y[train_indices], x[test_indices,:], y[test_indices]

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def predict_labels(x, w, method):

    assert method == "logistic" or "linear", "The model should be either logistic or linear"
    
    if method == 'linear':
        y_pred = x.dot(w)       
        y_pred[np.where(y_pred <= 0)] = -1
        y_pred[np.where(y_pred > 0)] = 1        
    elif method == 'logistic':
        y_pred = sigmoid(x.dot(w))       
        y_pred[np.where(y_pred <= 0.5)] = -1
        y_pred[np.where(y_pred > 0.5)] = 1 

    return y_pred

def compute_accuracy(y, x, w, method='linear'):
    y_pred = predict_labels(x, w, method)
    acc = np.mean(y_pred == y)

    return acc


def compute_loss(y, tx, w):
    """Calculate the loss.

    You can calculate the loss using mse or mae.
    """
    N = y.shape[0]
    loss = 0.5 * np.sum((y - tx.dot(w))**2, axis=0) / N
    
    return loss

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e = y - tx.dot(w)
    gradient = - tx.T.dot(e) / N
    
    return gradient

def least_squares(y, tx):
    w = np.linalg.inv(tx.T @ tx) @ tx.T @ y
    return w, compute_loss(y, tx, w)
    
def ridge_regression(y, tx, lambd):
    w = np.linalg.inv(tx.T @ tx + lambd * np.eye(np.shape(tx)[1])) @ tx.T @ y
    return w, compute_loss(y, tx, w)

def least_squares_GD(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y, tx, w)
        w = w - gamma*gradient
        
        ws.append(w)
        losses.append(loss)
        
        # if n_iter % 10 == 0:
        #     print("Gradient Descent({bi}/{ti}): loss={l}".format(
        #           bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]  

def least_squares_SGD(y, tx, initial_w, max_iters, gamma):
    """Stochastic Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        random_index = np.random.randint(y.shape[0], size=1)
        x = tx[random_index,:]
        y_sgd = y[random_index]
        
        loss = compute_loss(y, tx, w)
        gradient = compute_gradient(y_sgd, x, w)
            
        w = w - gamma*gradient
        
        if n_iter == int(max_iters/2):
            gamma = gamma/10
        
        ws.append(w)
        losses.append(loss)
        
        # if n_iter % 10 == 0:
        #     print("Stochastic Gradient Descent({bi}/{ti}): loss={l}".format(
        #           bi=n_iter, ti=max_iters - 1, l=loss))

    return ws[-1], losses[-1]

### logistic regression
def sigmoid(t):
    """apply the sigmoid function on t."""
    #out = np.exp(t) / (1+np.exp(t))
    out = 1 / (1 + np.exp(-t))
    
    return out

def logistic_regression(y, tx, initial_w, max_iters, gamma, method):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    y = y.copy()
    y[y==-1] = 0
    
    threshold = 1e-8
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    x = initial_w
    t = 1
    
    # start the logistic regression
    for iter in range(max_iters):
        # get loss and update w.
        
        if method == 'GD':
            w, loss = logistic_learning_by_gradient_descent(y, tx, w, gamma, 0)
        elif method == 'SGD':
            # gamma = 1/(iter+1)
            if iter == int(max_iters/2):
                gamma = gamma / 10
            w, loss = logistic_learning_by_stochastic_gradient_descent(y, tx, w, gamma, 0)
        elif method == 'AGD':
            loss, x_next = logistic_learning_by_gradient_descent(y, tx, w, gamma)
            t_next = 1/2 * (1 + np.sqrt(1 + 4 * np.square(t)))
            w = x_next + (t - 1) * (x_next - x) / t_next
            x = x_next
        elif method == 'Newton':
            w, loss = logistic_learning_by_newton_method(y, tx, w, gamma)        
        
        # log info
#         if iter % 10 == 0:
#             print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
        # converge criterion
        ws.append(w)
        losses.append(loss)
#         if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
#             break

    # print("loss={l}".format(l=losses[-1]))
                 
    return ws[-1], losses[-1]

def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma, method):
    y = y.copy()
    y[y==-1] = 0
    
    threshold = 1e-8
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    
    # start the logistic regression
    for iter in range(max_iters):
        
        if method == 'GD':
            # get loss and update w.
            w, loss = logistic_learning_by_gradient_descent(y, tx, w, gamma, lambda_)
            # log info
        elif method == 'SGD':
            if iter == int(max_iters/2):
                gamma = gamma/10
            w, loss = logistic_learning_by_stochastic_gradient_descent(y, tx, w, gamma, lambda_)                
            
#         if iter % 10 == 0:
#             print("Current iteration={i}, loss={l}".format(i=iter, l=loss))
            
        # converge criterion
        ws.append(w)        
        losses.append(loss)
#         if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
#             break

    # print("loss={l}".format(l=losses[-1]))
    return ws[-1], losses[-1]

def logistic_learning_by_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    N = y.shape[0]
    loss = np.mean(-y*tx.dot(w) + np.log(1+np.exp(tx.dot(w))), axis=0) + 0.5*lambda_* np.linalg.norm(w, 2)**2
    
    gradient = (tx.T.dot(sigmoid(tx.dot(w)) - y))/N + lambda_ * w
    w = w - gamma*gradient

    return w, loss

def logistic_learning_by_stochastic_gradient_descent(y, tx, w, gamma, lambda_):
    """
    Do one step of stochastic gradient descent using logistic regression.
    Return the loss and the updated w.
    """
    loss = np.mean(-y*tx.dot(w) + np.log(1+np.exp(tx.dot(w))), axis=0) + 0.5*lambda_* np.linalg.norm(w, 2)**2
    
    random_index = np.random.randint(y.shape[0], size=1)[0] 
    x_sto = tx[random_index,:]
    y_sto = y[random_index]
    gradient = x_sto.T.dot(sigmoid(x_sto.dot(w)) - y_sto) + lambda_ * w
    w = w - gamma*gradient

    return w, loss

def calculate_hessian(y, tx, w):
    """return the Hessian of the loss function."""
    S = np.diag(sigmoid(tx.dot(w)))
    hessian = tx.T.dot(S).tx
    
    return hessian

def logistic_learning_by_newton_method(y, tx, w, gamma):
    """
    Do one step on Newton's method.
    return the loss and updated w.
    """
    loss = calculate_logistic_regression_loss(y, tx, w, 0)
    gradient_first_order = calculate_logistic_gradient(y, tx, w, 0)
    gradient_second_order = calculate_hessian(y, tx, w)
    w = w - gamma * (np.linalg.inv(gradient_second_order).dot(gradient_first_order))
    return w, loss

# def logistic_learning_by_penalized_gradient(y, tx, w, gamma, lambda_):
#     """
#     Do one step of gradient descent, using the penalized logistic regression.
#     Return the loss and updated w.
#     """
#     loss = calculate_logistic_regression_loss(y, tx, w, lambda_)
#     gradient = calculate_logistic_gradient(y, tx, w, lambda_)
#     w = w - gamma*gradient
#     return w, loss

# def logistic_learning_by_penalized_stochastic_gradient_descent(y, tx, w, gamma, lambda_):
#     """
#     Do one step of gradient descent using logistic regression.
#     Return the loss and the updated w.
#     """
#     N = y.shape[0]
#     loss = calculate_logistic_regression_loss(y, tx, w, lambda_)
#     gradient = (tx.T.dot(sigmoid(tx.dot(w)) - y)) + lambda_ * w
#     w = w - gamma*gradient

#     return w, loss


def fea_augmentation(x, method, poly_degree=None, neg_poly_degree=None, concatenate=True, print_info=False, standarize=True):
    x_aug = x.copy() if concatenate else []
    
    if 'polynomial' in method:
        for d in range(2, poly_degree + 1):
            try:
                x_aug = np.c_[x_aug, x ** d]
            except:
                x_aug = x**d
    
    if 'cross_product_degree_2' in method:
        for i in range(x.shape[1]):
            for j in range(i+1, x.shape[1]):
                x_aug = np.c_[x_aug, x[:, i]*x[:, j]]
            if print_info:
                print('\rAdding degree-two cross product features | {:.2%} complete'.format(float((i+1)/x.shape[1])), end='', flush=True)
        print('')
        
    if 'cross_product_degree_3' in method:
        for i in range(x.shape[1]):
            if i != 1:
                x_aug = np.c_[x_aug, x[:, 1]*(x[:, i]**2)]
                x_aug = np.c_[x_aug, (x[:, 1]**2)*x[:, i]]

                if (i != 11):
                    x_aug = np.c_[x_aug, x[:, 11]*(x[:, i]**2)]
                    x_aug = np.c_[x_aug, (x[:, 11]**2)*x[:, i]]

                    if (i != 13):
                        x_aug = np.c_[x_aug, x[:, 13]*(x[:, i]**2)]
                        x_aug = np.c_[x_aug, (x[:, 13]**2)*x[:, i]]

                        if (i != 22):
                            x_aug = np.c_[x_aug, x[:, 22]*(x[:, i]**2)]
                            x_aug = np.c_[x_aug, (x[:, 22]**2)*x[:, i]]
            if print_info:
                print('\rAdding degree-three cross product features | {:.2%} complete'.format(float((i+1)/x.shape[1])), end='', flush=True)
        print('')
    
    if 'missing_value_indicator' in method:
        x_aug = np.c_[x_aug, indicator_feature_1]
        x_aug = np.c_[x_aug, indicator_feature_2]
    
    if 'neg_polynomial' in method:
        for d in range(1, neg_poly_degree+1):
            try:
                x_aug = np.c_[x_aug, 1 / (x**d + 1e-4)]
            except:
                x_aug = 1 / (x**d + 1e-4)
    
    x_aug_mean = x_aug.mean(axis=0)
    x_aug_std = x_aug.std(axis=0)
    
    # re-standarize to avoid numerical problems for gradient descent
    if standarize == True:
        x_aug = (x_aug - x_aug.mean(axis=0)) / x_aug.std(axis=0)
    
    # add a constant column to the features
    if concatenate:
        x_aug = np.c_[np.ones((len(x_aug), 1)), x_aug]

    return x_aug, x_aug_mean, x_aug_std


# functions related to cross-validation

def kf_split(X, y, folds=5, shuffle=True):
    """
    Split the dataset into 5 folds, and return 5 respective training and validation sets.
    The parameter 'shuffle' indicates if a random shuffle should be performed before splitting the data
    """
    if shuffle == True:
        index = np.arange(len(y))
        np.random.shuffle(index)
        X, y = X[index], y[index]

    fold_size = int(len(y) / folds)

    kf_split = []
    for i in range(folds):
        x_train = np.r_[X[:i * fold_size, :], X[(i + 1) * fold_size:, :]]
        y_train = np.r_[y[: i * fold_size], y[(i + 1) * fold_size:]]

        x_val = X[i * fold_size: (i + 1) * fold_size, :]
        y_val = y[i * fold_size: (i + 1) * fold_size]

        kf_split.append(((x_train, y_train), (x_val, y_val)))

    return kf_split


def least_squares_GD_CV(x, y, x_aug_temp=None, params_grid=None, max_iters=100, mode='GD'):
    """
    Perform cross validation for least squares (stochastic) gradient descent on the parameters grid.
    The 'mode' parameter indicates using GD or SGD
    """
    # retrieve parameter ranges
    lr_range = params_grid['lr']
    po_degree_range = params_grid['po_degree']
    ne_degree_range = params_grid['ne_degree']
    params_range = [(lr, po_degree, ne_degree) for lr in lr_range for po_degree in po_degree_range for ne_degree in ne_degree_range]

    # use 5-fold cross validation
    n_fold = 5
    fold_size = int(len(x) / n_fold)
    val_acc_mean = []

    stochastic_flag = '' if mode == 'GD' else 'S'
    print("Cross validation starts for least_squares_{}GD".format(stochastic_flag))
    print_table = PrettyTable(['lr', 'po_degree', 'ne_degree', 'val_accuracy'])

    # iterate over the parameters to find the optimal superparameters
    for i, (lr, po_degree, ne_degree) in enumerate(params_range):
        val_acc = []
        
        # do feature augmentation
        x_train_aug, _, _ = fea_augmentation(x, method = ['polynomial', 'neg_polynomial'], poly_degree=po_degree, neg_poly_degree=ne_degree, concatenate=False)
        x_train_aug = np.c_[x_aug_temp, x_train_aug]
        
        initial_w = np.random.normal(0, 0.01, (x_train_aug.shape[1], 1))

        # use 5-fold cross validation
        train_val_splits = kf_split(x_train_aug, y, folds=5, shuffle=True)

        for (x_train, y_train), (x_val, y_val) in train_val_splits:

            # choose GD or SGD accordingly
            if mode == 'GD':
                w, _ = least_squares_GD(y_train, x_train, initial_w, max_iters=max_iters, gamma=lr)
            elif mode == 'SGD':
                w, _ = least_squares_SGD(y_train, x_train, initial_w, max_iters=max_iters, gamma=lr)
            val_acc.append(compute_accuracy(y_val, x_val, w, method='linear'))

        # record the averaged val_accuracy on 5 folds
        val_acc_mean.append(sum(val_acc) / n_fold)

        # print validation result for each combination of superparameters
        print("\rfor lr: {}, po_degree : {}, ne_degree : {}, val_accuracy = {:.4f} | {:.2%} completed".format(lr, po_degree, ne_degree,\
                                                              float(sum(val_acc) / n_fold), (i + 1) / len(params_range)), end='', flush=True)
        print_table.add_row([lr, po_degree, ne_degree, '{:.4f}'.format(float(sum(val_acc) / n_fold))])
    
    # print the table containing validation accuracy for various parameter combinations
    print('\033[1m' + '\nLeast_Squares_{}GD cross validation result:'.format(stochastic_flag) + '\033[0m')
    print(print_table)

    # find the optimal parameters (with highest average validation accuracy)
    argmax_acc = [i for i, val in enumerate(val_acc_mean) if (val == max(val_acc_mean))][0]
    opt_param = params_range[argmax_acc]
    opt_val_acc = val_acc_mean[argmax_acc]

    # print the optimal parameters found by cross validation
    print('\033[1m' + "The optimal params for least squares {}GD are, lr: {}, po_degree: {}, ne_degree: {}, with val_accuracy: {}\n" \
          .format(stochastic_flag, opt_param[0], opt_param[1], opt_param[2], opt_val_acc) + '\033[0m')
    return opt_param


def least_squares_CV(x, y, x_aug_temp=None, params_grid=None):
    """
    Perform cross validation for least squares on the parameters grid.
    """
    # retrieve parameter range
    po_degree_range = params_grid['po_degree']
    ne_degree_range = params_grid['ne_degree']
    params_range = [(po_degree, ne_degree) for po_degree in po_degree_range for ne_degree in ne_degree_range]
    
    # use 5-fold cross validation
    n_fold = 5
    fold_size = int(len(x) / n_fold)
    val_acc_mean = []

    print("Cross validation starts for least_squares")
    print_table = PrettyTable(['po_degree', 'ne_degree', 'val_accuracy'])
    
    # iterate over the parameters to find the optimal superparameters
    for i, (po_degree, ne_degree) in enumerate(params_range):
        val_acc = []

        # do feature augmentation
        x_train_aug, _, _ = fea_augmentation(x, method = ['polynomial', 'neg_polynomial'], poly_degree=po_degree, neg_poly_degree=ne_degree, concatenate=False)
        x_train_aug = np.c_[x_aug_temp, x_train_aug]
        
        train_val_splits = kf_split(x_train_aug, y, folds=5, shuffle=True)
        for (x_train, y_train), (x_val, y_val) in train_val_splits:
            w, _ = least_squares(y_train, x_train)
            val_acc.append(compute_accuracy(y_val, x_val, w, method='linear'))

        # record the averaged val_accuracy on 5 folds
        val_acc_mean.append(sum(val_acc) / n_fold)

        # print validation result for each combination of superparameters
        print("\rfor po_degree : {}, 'ne_degree': {}, val_accuracy = {:.4f} | {:.2%} completed".format(po_degree, ne_degree, float(sum(val_acc) / n_fold),
                                                                                   (i + 1) / len(params_range)), end='', flush=True)
        print_table.add_row([po_degree, ne_degree, '{:.4f}'.format(float(sum(val_acc) / n_fold))])

    # print the table containing validation accuracy for various parameter combinations
    print('\033[1m' + '\nLeast_Squares cross validation result:' + '\033[0m')
    print(print_table)

    # find the optimal parameters (with highest average validation accuracy)
    argmax_acc = [i for i, val in enumerate(val_acc_mean) if (val == max(val_acc_mean))][0]
    opt_param = params_range[argmax_acc]
    opt_val_acc = val_acc_mean[argmax_acc]

    # print the optimal parameters found by cross validation
    print('\033[1m' + "The optimal params for least squares are, po_degree: {}, ne_degree: {}, with val_accuracy: {}\n".format(opt_param[0], opt_param[1], opt_val_acc) + '\033[0m')
    return opt_param


def ridge_regression_CV(x, y, x_aug_temp=None, params_grid=None):
    """
    Perform cross validation for least squares on the parameters grid.
    """
    # retrieve parameter ranges
    lambd_range = params_grid['lambd']
    po_degree_range = params_grid['po_degree']
    ne_degree_range = params_grid['ne_degree']
    params_range = [(lambd, po_degree, ne_degree) for lambd in lambd_range for po_degree in po_degree_range for ne_degree in ne_degree_range]

    # use 5-fold cross validation
    n_fold = 5
    fold_size = int(len(x) / n_fold)
    val_acc_mean = []

    print("Cross validation starts for ridge_regression")
    print_table = PrettyTable(['lambda', 'po_degree', 'ne_degree', 'val_accuracy'])

    # iterate over the parameters to find the optimal superparameters
    for i, (lambd, po_degree, ne_degree) in enumerate(params_range):
        val_acc = []
        
        # do feature augmentation
        x_train_aug, _, _ = fea_augmentation(x, method = ['polynomial', 'neg_polynomial'], poly_degree=po_degree, neg_poly_degree=ne_degree, concatenate=False)
        x_train_aug = np.c_[x_aug_temp, x_train_aug]
        
        train_val_splits = kf_split(x_train_aug, y, folds=5, shuffle=True)
        for (x_train, y_train), (x_val, y_val) in train_val_splits:
            w, _ = ridge_regression(y_train, x_train, lambd)
            val_acc.append(compute_accuracy(y_val, x_val, w, method='linear'))

        # record the averaged val_accuracy on 5 folds
        val_acc_mean.append(sum(val_acc) / n_fold)

        # print validation result for each combination of superparameters
        print("\rfor lambda: {}, po_degree : {}, ne_degree: {}, val_accuracy = {:.4f} | {:.2%} completed".format(lambd, po_degree, ne_degree, float(
            sum(val_acc) / n_fold), (i + 1) / len(params_range)), end='', flush=True)
        print_table.add_row([lambd, po_degree, ne_degree, '{:.4f}'.format(float(sum(val_acc) / n_fold))])

    # print the table containing validation accuracy for various parameter combinations
    print('\033[1m' + '\nRidge_regression cross validation result:' + '\033[0m')
    print(print_table)

    # find the optimal parameters (with highest average validation accuracy)
    argmax_acc = [i for i, val in enumerate(val_acc_mean) if (val == max(val_acc_mean))][0]
    opt_param = params_range[argmax_acc]
    opt_val_acc = val_acc_mean[argmax_acc]

    # print the optimal parameters found by cross validation
    print('\033[1m' + "The optimal params for ridge regression are, lambda: {}, po_degree: {}, ne_degree: {}, with val_accuracy: {}\n" \
          .format(opt_param[0], opt_param[1], opt_param[2], opt_val_acc) + '\033[0m')

    return opt_param

import gc


def logistic_regression_CV(x, y, x_aug_temp=None, params_grid=None, max_iters=100, mode='GD', reg=False):
    """
    Perform cross validation for logistic regression on the parameters grid.
    The 'mode' parameter indicates using GD or SGD
    The 'reg' parameter indicates using regularization or not
    """
    # retrieve the parameters (depending on using regularization or not, parameters can be different)
    lr_range = params_grid['lr']
    po_degree_range = params_grid['po_degree']
    ne_degree_range = params_grid['ne_degree']
    
    if reg == True:
        lambd_range = params_grid['lambd']
        params_range = [(lambd, lr, po_degree, ne_degree) for lambd in lambd_range for lr in lr_range for po_degree in po_degree_range for ne_degree in ne_degree_range]
        print_table = PrettyTable(['lambda', 'lr', 'po_degree', 'ne_degree', 'val_accuracy'])
    elif reg == False:
        params_range = [(lr, po_degree, ne_degree) for lr in lr_range for po_degree in po_degree_range for ne_degree in ne_degree_range]
        print_table = PrettyTable(['lr', 'po_degree', 'ne_degree', 'val_accuracy'])
    
    # use 5-fold cross validation
    n_fold = 5
    fold_size = int(len(x) / n_fold)
    val_acc_mean = []
    
    stochastic_flag = '' if mode == 'GD' else 'S'
    reg_flag = 'reg_' if reg == True else ''
    print("Cross validation starts for {}logistic_regression_{}GD".format(reg_flag, stochastic_flag))

    # iterate over the parameters to find the optimal superparameters
    for i, params in enumerate(params_range):
        
        gc.collect()

        if reg == True:
            lambd, lr, po_degree, ne_degree = params
        elif reg == False:
            lr, po_degree, ne_degree = params

        val_acc = []
        
        # do feature augmentation
        x_train_aug, _, _ = fea_augmentation(x, method = ['polynomial', 'neg_polynomial'], poly_degree=po_degree, neg_poly_degree=ne_degree, concatenate=False)
        x_train_aug = np.c_[x_aug_temp, x_train_aug]
        
        initial_w = np.zeros(x_train_aug.shape[1])
        
        train_val_splits = kf_split(x_train_aug, y, folds=5, shuffle=True)

        for (x_train, y_train), (x_val, y_val) in train_val_splits:

            # depending on regularization, use different algorithms, and indicate using GD or SGD
            if reg == True:
                w, _ = reg_logistic_regression(y_train, x_train, lambda_=lambd, initial_w=initial_w,
                                               max_iters=max_iters, gamma=lr, method=mode)
            elif reg == False:
                w, _ = logistic_regression(y_train, x_train, initial_w=initial_w, max_iters=max_iters, gamma=lr,
                                           method=mode)
        
            val_acc.append(compute_accuracy(y_val, x_val, w, method='logistic'))
        
        # record the averaged validation accuracy on 5 folds
        val_acc_mean.append(sum(val_acc) / n_fold)

        # print validation result for each combination of superparameters
        if reg == True:
            print("\rfor lambda: {}, lr: {}, po_degree : {}, ne_degree: {}, val_accuracy = {:.4f} | {:.2%} completed".format(lambd, lr, po_degree, ne_degree, float(sum(val_acc) / n_fold),(i + 1) / len(params_range)), end='', flush=True)
            print_table.add_row([lambd, lr, po_degree, ne_degree, '{:.4f}'.format(float(sum(val_acc) / n_fold))])
        elif reg == False:
            print("\rfor lr: {}, po_degree : {}, ne_degree: {}, val_accuracy = {:.4f} | {:.2%} completed".format(lr, po_degree, ne_degree, float(
                sum(val_acc) / n_fold), (i + 1) / len(params_range)), end='', flush=True)
            print_table.add_row([lr, po_degree, ne_degree, '{:.4f}'.format(float(sum(val_acc) / n_fold))])

            # print the table containing validation accuracy for various parameter combinations
    print('\033[1m' + '\n{}logistic_regression_{}GD validation result'.format(reg_flag, stochastic_flag) + '\033[0m')
    print(print_table)

    # find the optimal parameters (with highest average validation accuracy)
    argmax_acc = [i for i, val in enumerate(val_acc_mean) if (val == max(val_acc_mean))][0]
    opt_param = params_range[argmax_acc]
    opt_val_acc = val_acc_mean[argmax_acc]
    
    # print the optimal parameters found by cross validation

    if reg == True:
        print('\033[1m' + "The optimal params for regularized logistic regression ({}GD) are, lambda: {}, lr: {}, po_degree: {}, ne_degree: {}, with val_accuracy: {}\n" \
              .format(stochastic_flag, opt_param[0], opt_param[1], opt_param[2], opt_param[3], opt_val_acc) + '\033[0m')
    elif reg == False:
        print('\033[1m' + "The optimal params for logistic regression ({}GD) are, lr: {}, po_degree: {}, ne_degree: {}, with val_accuracy: {}\n" \
              .format(stochastic_flag, opt_param[0], opt_param[1], opt_param[2], opt_val_acc) + '\033[0m')
    return opt_param


def detect_missing_values(tX):
    """ This function detects and plots the percentage of missing values in each feature. """
    N, d = tX.shape
    miss_percentage = []
    for i in range(d):
        miss_percentage.append(len(tX[tX[:,i] == -999])/N)
        
    # plot the bar plot for missing percentages 
    plt.figure(figsize = (12, 5))
    plt.title('Percentage of missing values in each feature')
    plt.bar(np.arange(tX.shape[1]), miss_percentage)
    plt.xlabel('Index of feature')
    plt.ylabel('Percentage of missing values')
    plt.xticks(ticks=np.arange(tX.shape[1])) 
    plt.ylim(0, 1)
    plt.grid(axis='y')
    plt.show()
    
def plot_missing_value_class_ratio(tX, y):
    """ This function plots the positive/negative class ratio for the data points with missing value, and compare them with the overall positive/negative class ratio. """
    
    features_missing_value = [0, 4, 5, 6, 12, 23, 24, 25, 26, 27, 28]
    
    # for each feature, find the samples with missing values, and compute their ratio for positive samples/negative samples.
    missing_values_ratio = []
    for i in features_missing_value:
        missing_values_ratio.append(sum(tX[y==1,i]==-999)/sum(tX[y==-1,i]==-999))

    x = np.arange(len(features_missing_value))   # the label locations
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots(figsize = (12, 5))
    ax.bar(x, missing_values_ratio, width, color='r', label = 'ratio for the samples with missing value')

    ax.set_xlabel('Index of feature')
    ax.set_ylabel('Ratio')
    ax.set_title('Ratio between positive samples and negative samples')
    ax.set_xticks(x)
    ax.set_xticklabels(features_missing_value)
    ax.axhline(y=sum(y==1) / sum(y==-1), color="black", linestyle="--", label='overall ratio')
    
    plt.grid(axis='y')
    plt.legend(loc='lower right')
    plt.show()
    
def plot_distribution(tX, y):
    """ This function plots the data distribution of each feature. The distribution of samples with different labels are plotted seperately. """
    fig = plt.figure(figsize=(30, 25))
    
    for i in range(tX.shape[1]):
        tX_positive = tX[y==1, i]
        tX_negative = tX[y==-1, i]
        
        # plot the distributions
        ax = fig.add_subplot(6, 5, i+1) 
        n_positive, bins_positive, patches1 = ax.hist(tX_positive, bins=100, alpha = 0.5, label='positive')
        n_negative, bins_negative, patches2 = ax.hist(tX_negative, bins=bins_positive, alpha = 0.7, label='negative')

        ax.set_title('Distribution of feature {}'.format(i))
        ax.legend()
        
def clean_standarize(x, replace='median', mode='train_set', fea_means=None, fea_std=None, fea_replace=None):
    """ This function replaces the missing value with either median or mean value, and then normalizes the data. 
        It has two modes. 
        For 'training_set' mode, it calculates the mean, variance, replace value (mean or median) from the data.
        For 'test_set' mode, it gets the mean, variance, replace value directly from the parameters (calculated from the training set), and uses them to alter the test set.
    """
    if mode == 'train_set':
        fea_means = []
        fea_std = []
        fea_replace = [] # the value to be replaced with, either mean or median
        
        # record the mean and std from the training set
        for col in range(x.shape[1]):
            cleaned = list(filter(lambda x: x != -999, tX[:, col]))
            fea_means.append(np.mean(cleaned))
            fea_std.append(np.var(cleaned))
            if replace == 'median':
                fea_replace.append(np.median(cleaned))
        if replace == 'mean':
            fea_replace = fea_means
            
    # relace -999 by the median value (or mean)
    for col in range(x.shape[1]):
        for row in range(x.shape[0]):
            if x[row][col] == -999:
                x[row][col] = fea_replace[col]
    return (x - fea_means) / fea_std, fea_means, fea_std, fea_replace

def replace_missing_values(tX):
    """ This function relaces the missing values by median values """
    N, d = tX.shape
    tX_replace = tX.copy()
    
    for i in range(d):
        n_missing = len(tX_replace[tX_replace[:,i] == -999])        
        if n_missing > 0:
            tX_replace[np.where(np.array(tX_replace) == -999)] = np.median(tX_replace[tX_replace[:,i] != -999])
    
    return tX_replace   

def train_and_test(data, method):
    """ This function trains and tests on the data, using either least square or ridge. """
    (x_train, y_train), (x_test, y_test) = data
    
    if method == 'least square':
        w, _  = least_squares(y_train, x_train)
    elif method == 'ridge':
        w, _ = ridge_regression(y_train, x_train, 0.00001)
    
    train_acc, test_acc = \
        compute_accuracy(y_train, x_train, w, method='linear'), compute_accuracy(y_test, x_test, w, method='linear')
    
    return (train_acc, test_acc)

jet_num=tX[:, 22]

def expand_over_jetnum(x):
    """ This function expands the features over the jet_num feature. The details of this expansion is discussed in the markdown cells. """
    x_expand = np.zeros((x.shape[0], 4*x.shape[1]))
    for n, j_n in enumerate(jet_num):
        x_expand[n, int(j_n*x.shape[1]):int((j_n+1)*x.shape[1])] = x[n, :]
    return x_expand

def one_hot_encoder(column):
    """ This function translates a column into one-hot encodings. """
    categories = np.unique(column)
    one_hot_codes = np.eye(categories.shape[0])
    
    one_hot_labels = []
    for i in column:
        one_hot_label = one_hot_codes[int(i)]
        one_hot_labels.append(one_hot_label)
     
    return np.array(one_hot_labels) 

def PCA(x, components):
    """ This function computes the mean and PCA basis """
    mean = np.mean(x)
    b = x - mean
    Sigma = b.T@b
    eigen_values, eigen_vectors = np.linalg.eig(Sigma)

    right_order = np.argsort(np.abs(eigen_values))[::-1]
    basis = eigen_vectors[:, right_order]
    
    output = x.dot(basis[:,:components])
    
    return output, basis[:,:components]