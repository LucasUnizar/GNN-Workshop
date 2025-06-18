import numpy as np
import torch


def rrmse_inf(data_ground_truth, data_predicted):
    """
    Calculate the Root Mean Squared Error (RMSE) normalized by the infinity norm.

    Please pay attention to the inputs, it must have the following shape:
        (snapshots, nodes, variables).
        E.g. (200, 550, 3) would be 200 snapshots, of a graph of 550 nodes and 3 variables, such as position.
        E.g. (200, 550, 1) would be 200 snapshots, of a graph of 550 nodes and 1 variable, such as S.Mises.
        Eg (550, 3) would be taken care of automatically and transformed as (1, 550, 3)
        E.g. (200, 550) WON'T WORK, because it will interpretate the input as 550 variable vector.

    Args:
        data_ground_truth (numpy.ndarray): Ground truth data with shape (snapshot, nodes, variable).
                                           In case of (nodes, variable), it expands snapshot dim to 1.
        data_predicted (numpy.ndarray): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variable), it expands snapshot dim to 1.

    Returns:
        float: RMSE normalized by the infinity norm.
    """

    return np.sqrt(mse_inf(data_ground_truth, data_predicted))


def mse_inf(data_ground_truth, data_predicted):
    """
    Calculate the Mean Squared Error (MSE) normalized by the infinity norm.

    Please pay attention to the inputs, it must have the following shape:
        (snapshots, nodes, variables).
        E.g. (200, 550, 3) would be 200 snapshots, of a graph of 550 nodes and 3 variables, such as position.
        E.g. (200, 550, 1) would be 200 snapshots, of a graph of 550 nodes and 1 variable, such as S.Mises.
        Eg (550, 3) would be taken care of automatically and transformed as (1, 550, 3)
        E.g. (200, 550) WON'T WORK, because it will interpretate the input as 550 variable vector.

    Args:
        data_ground_truth (numpy.ndarray): Ground truth data with shape (snapshot, nodes, variable).
                                           In case of (nodes, variable), it expands snapshot dim to 1.
        data_predicted (numpy.ndarray): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variable), it expands snapshot dim to 1.

    Returns:
        float: RMSE normalized by the infinity norm.
    """

    x, y = data_ground_truth, data_predicted

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)
        y = np.asarray(y)

    if len(x.shape) == 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

    # Infinity norm calculation
    se_inf = []
    for i in range(x.shape[0]):
        x_snap = x[i]
        error = x[i] - y[i]
        infinite_norm_se = np.linalg.norm(x_snap, ord=np.inf) ** 2 + 1e-6
        for j in range(x_snap.shape[0]):
            l2_norm_se = np.linalg.norm(error[j]) ** 2
            se_inf.append(l2_norm_se / infinite_norm_se)

    return np.mean(np.array(se_inf))


def se_inf(data_ground_truth, data_predicted):
    """
    Calculate the Mean Squared Error (MSE) normalized by the infinity norm.

    Please pay attention to the inputs, it must have the following shape:
        (snapshots, nodes, variables).
        E.g. (200, 550, 3) would be 200 snapshots, of a graph of 550 nodes and 3 variables, such as position.
        E.g. (200, 550, 1) would be 200 snapshots, of a graph of 550 nodes and 1 variable, such as S.Mises.
        Eg (550, 3) would be taken care of automatically and transformed as (1, 550, 3)
        E.g. (200, 550) WON'T WORK, because it will interpretate the input as 550 variable vector.

    Args:
        data_ground_truth (numpy.ndarray): Ground truth data with shape (snapshot, nodes, variable).
                                           In case of (nodes, variable), it expands snapshot dim to 1.
        data_predicted (numpy.ndarray): Predicted data with the same shape as data_ground_truth.
                                        In case of (nodes, variable), it expands snapshot dim to 1.

    Returns:
        float: RMSE normalized by the infinity norm.
    """

    x, y = data_ground_truth, data_predicted

    if isinstance(x, torch.Tensor):
        x = np.asarray(x)
        y = np.asarray(y)

    if len(x.shape) == 2:
        x = np.expand_dims(x, 0)
        y = np.expand_dims(y, 0)

    # Infinity norm calculation
    se_inf_list = []
    for i in range(x.shape[0]):
        x_snap = x[i]
        error = x[i] - y[i]
        infinite_norm_se = np.linalg.norm(x_snap, ord=np.inf) ** 2 + 1e-6
        for j in range(x_snap.shape[0]):
            l2_norm_se = np.linalg.norm(error[j]) ** 2
            se_inf_list.append(l2_norm_se / infinite_norm_se)

    return np.array(se_inf_list)

def mse(x, y):
    """
    Calculates the Mean Squared Error (RMSE) between two arrays x and y.

    Parameters:
    x (array-like): Array of true values.
    y (array-like): Array of predicted values.

    Returns:
    float: The Root Mean Squared Error (RMSE) between x and y.
    """
    # Calculate squared error
    se = (x - y) ** 2

    return se.mean()

def mse_roll(x, y):
    """
    Calculates the Mean Squared Error (RMSE) between two arrays x and y.

    Parameters:
    x (array-like): Array of true values.
    y (array-like): Array of predicted values.

    Returns:
    float: The Root Mean Squared Error (RMSE) between x and y.
    """
    # Calculate squared error
    se = (x - y) ** 2

    return se.mean(1)

def rmse(x, y):
    """
    Calculates the Root Mean Squared Error (RMSE) between two arrays x and y.

    Parameters:
    x (array-like): Array of true values.
    y (array-like): Array of predicted values.

    Returns:
    float: The Root Mean Squared Error (RMSE) between x and y.
    """
    # Calculate squared error
    se = (x - y) ** 2

    return se.mean() ** 0.5