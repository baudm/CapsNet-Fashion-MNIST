#!/usr/bin/env python

from keras.datasets import fashion_mnist
import numpy as np


def draw(canvas, image):
    x_off = np.random.randint(0, 15)
    y_off = np.random.randint(0, 15)
    w, h = image.shape
    for x in range(w):
        for y in range(h):
            j = image[x][y]
            if j > 64:
                canvas.itemset((x + x_off, y + y_off), j)


def sample(x_pool, y_pool):
    n = x_pool.shape[0]
    first = second = np.random.randint(n)
    # Avoid multiple instances of the same class
    while y_pool[first] == y_pool[second]:
        second = np.random.randint(n)
    x1 = x_pool[first]
    x2 = x_pool[second]
    combined = np.zeros((42, 42), dtype='uint8')
    draw(combined, x1)
    draw(combined, x2)
    return combined, [first, second]


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

import os.path
import uuid


def combine(outdir, x_pool, y_pool, num_samples, bundle_size):
    x_all = []
    idx_all = []
    while num_samples:
        x, idx = sample(x_pool, y_pool)
        x_all.append(x)
        idx_all.append(idx)
        num_samples -= 1
        if num_samples % 10000 == 0:
            print(num_samples)
        if len(x_all) >= bundle_size:
            x_all = np.stack(x_all)
            idx_all = np.stack(idx_all)
            fname = 'cfmnist_' + str(uuid.uuid4()) + '.npz'
            fname = os.path.join(outdir, fname)
            print(fname)
            with open(fname, 'wb') as f:
                np.savez(f, x=x_all, idx=idx_all)
            x_all = []
            idx_all = []

import sys

def main():
    outdir = sys.argv[1]
    num_samples = 600000
    num_test = 10000000
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    combine(outdir, x_train, y_train, num_samples, 100000)
    #combine(x_test, y_test, num_test)
        

if __name__ == '__main__':
    main()
