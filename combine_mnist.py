#!/usr/bin/env python

from capsulenet import load_mnist
import numpy as np


def draw(canvas, image):
    x_off = np.random.randint(0, 13)
    y_off = np.random.randint(0, 13)
    w, h, d = image.shape
    for x in range(w):
        for y in range(h):
            j = image[x][y]
            if j > 64:
                canvas.itemset((x + x_off, y + y_off), j)


def sample_and_combine(x_pool, y_pool):
    n = x_pool.shape[0]
    first = second = np.random.randint(n)
    while second == first:
        second = np.random.randint(n)
    x1 = x_pool[first]
    y1 = y_pool[first]
    x2 = x_pool[second]
    y2 = y_pool[second]
    combined = np.zeros((40, 40))
    draw(combined, x1)
    draw(combined, x2)
    y1[np.argmax(y2)] = 1
    return combined, y1

import uuid


def combine(x_pool, y_pool, num_samples):
    c_x_train = []
    c_y_train = []
    while num_samples:
        x, y = sample_and_combine(x_pool, y_pool)
        c_x_train.append(x)
        c_y_train.append(y)
        num_samples -= 1
        if len(c_x_train) >= 1000000:
            c_x_train = np.stack(c_x_train)
            c_y_train = np.stack(c_y_train)
            fname = 'cfmnist_' + str(uuid.uuid4())
            print(fname)
            with open(fname, 'wb') as f:
                np.savez(f, x=c_x_train, y=c_y_train)
            c_x_train = []
            c_y_train = []


def main():
    num_samples = 60000000
    num_test = 10000000
    (x_train, y_train), (x_test, y_test) = load_mnist()
    combine(x_train, y_train, num_samples)
    combine(x_test, y_test, num_test)
        

if __name__ == '__main__':
    main()
