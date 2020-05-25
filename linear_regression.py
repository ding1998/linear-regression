import torch
from torch import autograd
import numpy as np


def compute_average_loss(b, w, points):
    total_error = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        total_error += ((w * x + b) - y) ** 2
    return total_error / len(points)


def autograd_update_W_B(b_current, w_current, points, lr):
    b_gradent = 0
    w_gradent = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradent += -(2 / N )* (y-(w_current * x + b_current))
        w_gradent += -(2 / N) * (y-(w_current * x + b_current) )* x
    new_b = b_current - b_gradent * lr
    new_w = w_current - w_gradent * lr
    return [new_b, new_w]


def gradients_runnner(b_start, w_start, lr, Num_iterations, points):
    b = b_start
    w = w_start
    for i in range(Num_iterations):
        b, w = autograd_update_W_B(b, w, np.array(points), lr)
    return [b, w]


def run():
    points = np.genfromtxt("data.csv", delimiter=",")
    lr = 0.0001
    Num_iterations = 1000
    b_start = 0
    w_start = 0
    print("Starting gradient descent at b ={0},w={1},error={2}".format(b_start, w_start,
                                                                       compute_average_loss(b_start, w_start, points)))

    print("Running.....")
    [b, w] = gradients_runnner(b_start, w_start, lr, Num_iterations, points)

    print("after {0} iterations w={1},b={2},error={3}".format(Num_iterations, w, b, compute_average_loss(b, w, points)))


if __name__ == '__main__':
    run()
