"""
Created on 2018-04-16
Project: MRC-3vG-CARIC
File: data_utils_3vG
...
@author: Alvin(Xinyao) Sun
"""
import os

import matplotlib.pyplot as plt
import numpy as np

def readShortComplex(fileName, width=1):
    return np.fromfile(fileName, '>i2').astype(float).view(complex).reshape(-1, width)


def readFloatComplex(fileName, width=1):
    return np.fromfile(fileName, '>c8').astype(complex).reshape(-1, width)


def readFloat(fileName, width=1):
    return np.fromfile(fileName, '>f4').astype(float).reshape(-1, width)


def writeShortComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.copy().view(float).astype('>i2').tofile(out_file)
    out_file.close()


def writeFloatComplex(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>c8').tofile(out_file)
    out_file.close()


def writeFloat(fileName, data):
    out_file = open(fileName, 'wb')
    data.astype('>f4').tofile(out_file)
    out_file.close()


# %%
def readFloatComplexRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            size_of_file = os.path.getsize(fileName)
            height = size_of_file / 8 / width
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(8 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(8 * patch_size), dtype=">c8").astype(complex))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height

def readShortFloatComplexRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            size_of_file = os.path.getsize(fileName)
            # print(size_of_file)
            height = size_of_file / 4 / width
            # print(height)
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(4 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(4 * patch_size), dtype=">i2").astype(float).view(complex))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height

def readFloatRandomPathces(fileName, width=1, num_sample=1, patch_size=1, rows=None, cols=None, height=None):
    with open(fileName, "rb") as fin:
        if rows is None:
            rows = np.random.randint(0, high=(height - patch_size), size=num_sample)
            cols = np.random.randint(0, high=(width - patch_size), size=num_sample)
            size_of_file = os.path.getsize(fileName)
            height = size_of_file / 4 / width
        patches = []
        for i in range(len(rows)):
            row = rows[i]
            col = cols[i]
            img = []
            for p_row in range(patch_size):
                fin.seek(4 * (width * (row + p_row) + col))
                img.append(np.frombuffer(fin.read(4 * patch_size), dtype=">f4").astype(float))
            patches.append(np.reshape(img, [patch_size, patch_size]))
    return patches, rows, cols, height


