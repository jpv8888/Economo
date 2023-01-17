# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:25:39 2023

@author: jpv88
"""

import matplotlib

import JV_utils

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from os import listdir
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from tqdm import tqdm

path = r'C:/Users/jpv88/OneDrive/Documents/GitHub/Economo/HCR analysis/Data/'
files = onlyfiles = [f for f in listdir(path)]

barcodes = []
for file in files:

    anno = pd.read_csv(path + file)
    
    xyz = anno[["X", "Y", "Z"]]
    anno.drop(columns=['X', 'Y', 'Z', 'GFP', 'Snap25_1', 'Snap25_2', 'Snap25_3', 
                       'Snap25_5', 'Snap25_6'], inplace=True)
    
    anno_array = anno.to_numpy()
    barcodes.append(anno_array)

genes = anno.columns.tolist()
    
barcodes = np.concatenate(barcodes)    

pca = PCA(n_components=2)

transform = pca.fit_transform(barcodes)

plt.scatter(transform[:,0], transform[:,1])

kmeans = KMeans(n_clusters=5).fit(transform)

plt.scatter(transform[:,0], transform[:,1], c=kmeans.labels_)

# %%

n_clusters = np.arange(1, 10, 1)
distance_pens = []

for n in n_clusters:
    kmeans = KMeans(n_clusters=n).fit(barcodes)
    distance = kmeans.transform(barcodes)
    labels = kmeans.labels_
    distance_pen = 0
    for i in range(len(labels)):
        distance_pen += distance[i,labels[i]]
    distance_pens.append(distance_pen)

# %%

import umap.umap_ as umap

reducer = umap.UMAP(n_neighbors=3)

embedding = reducer.fit_transform(barcodes)

plt.scatter(embedding[:,0], embedding[:,1], c=barcodes[:,10])

# %%

from sklearn.ensemble import RandomForestClassifier

importances = []
for _ in tqdm(range(10)):
    n_clusters = np.arange(2, 10, 1)
    for n in n_clusters:
        kmeans = KMeans(n_clusters=n).fit(barcodes)
        labels = kmeans.labels_
        forest = RandomForestClassifier()
        forest.fit(barcodes, labels)
        importances.append(forest.feature_importances_)
    
importances = np.stack(importances)
imp_means = np.mean(importances, axis=0)

cmap = mpl.cm.viridis
colors = imp_means/max(imp_means)

fig, ax = plt.subplots()
plt.bar(genes, imp_means, color=cmap(colors))
plt.errorbar(genes, imp_means, yerr=np.std(importances, axis=0), fmt='none', 
             color="k", capsize=2)
plt.ylabel('Mean Decrease in Impurity (MDI)', fontsize=13)
plt.title('Relative Importance of Gene in Random Forest Classification',  
          fontsize=13)

plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right', 
         rotation_mode="anchor")

plt.yticks(fontsize=14)
plt.xticks(fontsize=11)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()

plt.ylim(0)


# %% total expression level per gene


cmap = mpl.cm.viridis

reads = np.sum(barcodes, axis=0)
reads = reads/len(barcodes)
reads = reads.tolist()

genes_sorted = JV_utils.sort_list_by_list(reads, genes)
reads = sorted(reads)

genes_sorted = list(reversed(genes_sorted))
reads = list(reversed(reads))

fig, ax = plt.subplots()

colors = np.array(reads)*100/100
bars = plt.bar(genes_sorted, np.array(reads)*100, zorder=2, edgecolor='k', 
               linewidth=2, color=cmap(colors))
plt.setp(ax.get_xticklabels(), rotation=40, horizontalalignment='right', 
         rotation_mode="anchor")

x = []
for bar in bars:
    x.append(bar.xy[0])

plt.yticks(fontsize=14)
plt.xticks(fontsize=11)
plt.ylabel('% Expression', fontsize=16)
plt.title('Expression Levels in GFP Tagged Neurons', fontsize=18)
plt.grid(axis='y', which='both', ls='--', alpha=1, lw=0.2, zorder=1)

plt.ylim(0, 100)

# Create offset transform by 5 points in x direction
dx = 5/72.; dy = 0/72. 
offset = matplotlib.transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
for label in ax.xaxis.get_majorticklabels():
    label.set_transform(label.get_transform() + offset)
    
    
plt.tight_layout()



# %%

import random

# probability of expression
def flip(p):
    return 1 if random.random() < p else 0

def coexpress(p1, p2, n_iters=1000):
    
    hits = 0
    for _ in range(n_iters):
        gene1 = flip(p1)
        gene2 = flip(p2)
        if (gene1 == 1) and (gene2 == 1):
            hits += 1
            
    return hits/n_iters

def coexpress_data(barcodes, gene1_idx, gene2_idx,
                   bootstrap_len=np.shape(barcodes)[0]):
    
    n_cells = np.shape(barcodes)[0]
    cell_idxs = list(range(n_cells))
    idxs = random.choices(cell_idxs, k=bootstrap_len)
    
    hits = 0
    for idx in idxs:
        gene1 = barcodes[idx,gene1_idx]
        gene2 = barcodes[idx,gene2_idx]
        if (gene1 == 1) and (gene2 == 1):
            hits += 1
    
    return hits/bootstrap_len

def coexpress_true(barcodes, gene1_idx, gene2_idx):
    
    n_cells = np.shape(barcodes)[0]
    
    hits = 0
    for idx in range(n_cells):
        gene1 = barcodes[idx,gene1_idx]
        gene2 = barcodes[idx,gene2_idx]
        if (gene1 == 1) and (gene2 == 1):
            hits += 1
    
    return hits/n_cells

def find_gene_idx(genes, gene_subset):
    gene_idxs = []
    for gene in gene_subset:
        gene_idxs.append(np.where(np.array(genes) == gene)[0][0])
        
    return gene_idxs
    

import numpy as np
import scipy.stats

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

# %%

reads = np.sum(barcodes, axis=0)
reads = reads/len(barcodes)
reads = reads.tolist()

gene_subset = ['Tac1', 'Gal']
gene_idxs = []
for gene in gene_subset:
    gene_idxs.append(np.where(np.array(genes) == gene)[0][0])

p1 = reads[gene_idxs[0]]
p2 = reads[gene_idxs[1]]

co_null = []
co_true = []
for _ in range(10000):
    co_null.append(coexpress(p1, p2, n_iters=len(barcodes)))
    
p_value = stats.ttest_ind(co_true, co_null)[1]

co_true = coexpress_true(barcodes, gene_idxs[0], gene_idxs[1])

norm_dist = stats.norm.pdf(co_true, loc=np.mean(co_null), scale=np.std(co_null))

# %% 

import itertools
from tqdm import tqdm

reads = np.sum(barcodes, axis=0)
reads = reads/len(barcodes)
reads = reads.tolist()

gene_subsets = itertools.combinations(genes, 2)
gene_subsets = list(gene_subsets)

p_vals = []
directions = []
for gene_subset in tqdm(gene_subsets):
    
    gene_idxs = []
    for gene in gene_subset:
        gene_idxs.append(np.where(np.array(genes) == gene)[0][0])
    
    p1 = reads[gene_idxs[0]]
    p2 = reads[gene_idxs[1]]
    
    co_null = []
    for _ in range(10000):
        co_null.append(coexpress(p1, p2, n_iters=len(barcodes)))
    
    co_true = coexpress_true(barcodes, gene_idxs[0], gene_idxs[1])
    
    if co_true < np.mean(co_null):
        p_vals.append(2*stats.norm.cdf(co_true, loc=np.mean(co_null), 
                                       scale=np.std(co_null)))
        directions.append('low')
        
    elif co_true >= np.mean(co_null):
        p_vals.append(2*stats.norm.sf(co_true, loc=np.mean(co_null), 
                                      scale=np.std(co_null)))
        directions.append('high')

# %%

alpha = 0.05

p_vals = np.array(p_vals)
sigs = (p_vals < alpha)

directions = np.array(directions)
pos = (directions == 'high')

gene_subsets = np.array(gene_subsets)
edges = gene_subsets[pos & sigs]

# %%

import networkx as nx
G = nx.Graph()

# for edge in edges:
#     idxs = find_gene_idx(genes, edge)
#     G.add_edge(idxs[0], idxs[1])
    
for edge in edges:
    G.add_edge(*edge)
    
margin = 1000
    
subax1 = plt.subplot()
pos = nx.spring_layout(G, scale=2)

pos_list = []
for val in pos.values():
    pos_list.append(val)
pos_array = np.stack(pos_list)

nx.draw_networkx_nodes(G, pos, node_size=2000, alpha=0.2)
nx.draw_networkx_edges(G, pos, min_source_margin=margin, min_target_margin=margin)
nx.draw_networkx_labels(G, pos, font_size=18, font_weight='bold', font_color='black',
                        bbox=dict(boxstyle='circle', fc="dodgerblue", ec="k"))

plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.tight_layout()




    
    
    
    


