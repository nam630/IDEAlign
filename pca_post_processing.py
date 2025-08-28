import numpy as np
import pickle
import pandas as pd
import os
from sklearn.decomposition import PCA
import math 
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
from scipy.stats import rankdata
from itertools import combinations


f = pickle.load(open("reasoning_human-eval_scores.pkl", "rb"))
human_eval = np.array(f).mean(axis=0)

def pca_projected():
    for file in files:
        print("First need to generate the embeddings using embed.py")
        # Load the embedding 
        X = pickle.load(open(file,"rb"))
        dim = X.shape[-1]
        inputs = X.reshape(-1, dim)
        inputs = inputs - inputs.mean(axis=0).reshape(1, -1)
        pca = PCA()
        pca.fit(inputs)
        components = pca.components_
        # remove the top components
        d = math.ceil(dim / 100)
        reduced = np.zeros(inputs.shape)
        for i in range(d):
            reduced += (inputs @ components[i]).reshape(-1, 1) @ components[i].reshape(1, -1)
        processed = inputs - reduced
        outputs = processed.reshape(4, 15, dim)
        model_eval = np.zeros((15, 15))
        for i in range(4):
            model_eval += cosine_similarity(outputs[i])
        # compare model_eval with human_eval based on the produced rankings
        i_lower = np.tril_indices(15, k=-1)
        human_scores = human_eval[i_lower]
        model_scores = model_eval[i_lower]
        coeff = np.corrcoef(human_scores, model_scores)
        print(file)
        print("Avg correlation coeff: ", coeff[0,1],"\n")
        human_ranks = rankdata(human_scores, method='average')
        model_ranks = rankdata(model_scores, method='average')
        rho, p_value = spearmanr(human_ranks, model_ranks)
        print("spearman ", rho, p_value)

def pca_reduced():
    for file in files:
        X = pickle.load(open(file,"rb"))
        dim = X.shape[-1]
        inputs = X.reshape(-1, dim)
        inputs = inputs - inputs.mean(axis=0).reshape(1, -1)
        pca = PCA(n_components=math.ceil(2*len(inputs)/3))
        processed = pca.fit_transform(inputs)
        outputs = processed.reshape(4, 15, -1)
        model_eval = np.zeros((15, 15))
        for i in range(4):
            model_eval += cosine_similarity(outputs[i])
        # compare model_eval with human_eval based on the produced rankings
        i_lower = np.tril_indices(15, k=-1)
        human_scores = human_eval[i_lower]
        model_scores = model_eval[i_lower]
        coeff = np.corrcoef(human_scores, model_scores)
        print(file)
        print("Avg correlation coeff: ", coeff[0,1],"\n")
        human_ranks = rankdata(human_scores, method='average')
        model_ranks = rankdata(model_scores, method='average')
        rho, p_value = spearmanr(human_ranks, model_ranks)
        print("spearman ", rho, p_value)

if __name__ == "__main__":
    pca_projected()
    pca_reduced()
