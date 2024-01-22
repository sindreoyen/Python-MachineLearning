#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# CONJUNTOS DE DATOS A USAR EN EL PRIMER TRABAJO DE LA ASIGNATURA 
# "APRENDIZAJE AUTOMÁTICO"

import numpy as np
import os


# ----------------------------------------------------

# CONCESIÓN DE UN PRÉSTAMO

from .datos import credito

X_credito=np.array([d[:-1] for d in credito.datos_con_clas])
y_credito=np.array([d[-1] for d in credito.datos_con_clas])

# ----------------------------------------------------

# CLASIFICACIÓN DE LA PLANTA DE IRIS

from sklearn.datasets import load_iris

iris=load_iris()
X_iris=iris.data
y_iris=iris.target


# --------------------------------------------------

# VOTOS EN EL CONGRESO USA

from .datos import votos
X_votos=votos.datos
y_votos=votos.clasif



#--------------------------------------------------

# CÁNCER DE MAMA

from sklearn.datasets import load_breast_cancer

cancer=load_breast_cancer()

X_cancer=cancer.data
y_cancer=cancer.target

#-------------------------------------------

# CRÍTICAS DE PELÍCULAS EN IMDB

# Los datos están obtebidos de esta manera

#import random as rd
#from sklearn.datasets import load_files
#
#reviews_train = load_files("datos/aclImdb/train/")
#muestra_entr=rd.sample(list(zip(reviews_train.data,reviews_train.target)),k=2000)
#text_train=[d[0] for d in muestra_entr]
#text_train = [doc.replace(b"<br />", b" ") for doc in text_train]
#y_train=np.array([d[1] for d in muestra_entr])
#print("Ejemplos por cada clase: {}".format(np.bincount(y_train)))
#
#reviews_test = load_files("datos/aclImdb/test/")
#muestra_test=rd.sample(list(zip(reviews_test.data,reviews_test.target)),k=400)
#text_test=[d[0] for d in muestra_test]
#text_test = [doc.replace(b"<br />", b" ") for doc in text_test]
#y_test=np.array([d[1] for d in muestra_test])
#print("Ejemplos por cada clase: {}".format(np.bincount(y_test)))
#
#
#from sklearn.feature_extraction.text import CountVectorizer
#
#vect = CountVectorizer(min_df=50, stop_words="english").fit(text_train)
#print("Tamaño del vocabulario: {}".format(len(vect.vocabulary_)))
#X_train = vect.transform(text_train).toarray()
#X_test = vect.transform(text_test).toarray()
#
#np.save("datos/imdb_sentiment/vect_train_text.npy",X_train)
#np.save("datos/imdb_sentiment/vect_test_text.npy",X_test)
#np.save("datos/imdb_sentiment/y_train_text.npy",y_train)
#np.save("datos/imdb_sentiment/y_test_text.npy",y_test)

# Find the path to the current file
file_path = os.path.abspath(os.path.dirname(__file__))

imdb_sentiment_path = os.path.join(file_path, "datos", "imdb_sentiment")
X_train_imdb=np.load(os.path.join(imdb_sentiment_path, "vect_train_text.npy"))
X_test_imdb=np.load(os.path.join(imdb_sentiment_path, "vect_test_text.npy"))
y_train_imdb=np.load(os.path.join(imdb_sentiment_path, "y_train_text.npy"))
y_test_imdb=np.load(os.path.join(imdb_sentiment_path, "y_test_text.npy"))

# ----------------------------------------------------------------

# DÍGITOS ESCRITOS A MANO

# Path to the data directory
digits_path = os.path.join(file_path, "datos", "digitdata")

# The digits dataset
path_x_train = os.path.join(digits_path, "trainingimages")
path_y_train = os.path.join(digits_path, "traininglabels")
path_x_test = os.path.join(digits_path, "testimages")
path_y_test = os.path.join(digits_path, "testlabels")
path_x_valid = os.path.join(digits_path, "validationimages")
path_y_valid = os.path.join(digits_path, "validationlabels")

def print_digit(digit):
    for line in digit:
        print("".join([{1: "#", 0: " "}[c] for c in line]))

def load_digits(path_x: str, path_y: str, digit_height=28):
    with open(path_x) as file_x, open(path_y) as file_y:
        y = [int(line.rstrip()) for line in file_y]
        X = [[[{"#": 1, "+": 1, " ": 0}[c] for c in file_x.readline().rstrip()] for _ in range(digit_height)] for _ in range(len(y))]
        for i, x in enumerate(X):
            for j, line in enumerate(x):
                X[i][j] += [0] * (digit_height - len(line))
    return np.array(X, dtype=float), np.array(y, dtype=int)

X_train_digits, y_train_digits = load_digits(path_x_train, path_y_train)
X_test_digits, y_test_digits = load_digits(path_x_test, path_y_test)
X_valid_digits, y_valid_digits = load_digits(path_x_valid, path_y_valid)