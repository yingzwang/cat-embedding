# visualize embeddings in a 2D plot

import pickle
from models import load_embeddings
from sklearn import manifold
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import numpy as np
import pandas as pd
import os

# directory of the pickle files
data_dir = 'data/'

# directory for saving the images
image_dir = 'results/'

# load embeddings
em_name = 'embeddings_auto.pickle' # auto calculated (half of the original dim, at most 50)
#em_name = 'embeddings_ref.pickle' # paper ver
embeddings_path = os.path.join(data_dir, em_name)
features_em, embeddings_dict, em_size = load_embeddings(embeddings_path)

# load LabelEncoders
with open(os.path.join(data_dir, "les.pickle"), 'rb') as f:
    les_dict = pickle.load(f) # usage: les_dict['DayOfWeek']
print("label encoded features: ", les_dict.keys())


def plot_2D(xx, yy, names, figsize=(8, 8)):
    # plot 2D results
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(xx, yy, 'o', markeredgecolor='k', alpha=0.6, markersize=10)
    for i, txt in enumerate(names):
        ax.annotate(txt, (xx[i], yy[i]), xytext=(8.5, -5), textcoords='offset points')
    ax.axis('equal')
    # Hide the right and top spines
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # Only show ticks on the left and bottom spines
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    
    
def plot_save_tsne(embedding_vec, names, fig_path, figsize=(8, 8)):
    """
    tsne plot of the embedding vectors,
    and save the figure
    """
    tsne = manifold.TSNE(init='pca', random_state=0, method='exact')
    Y = tsne.fit_transform(embedding_vec)
    xx, yy = -Y[:, 0], -Y[:, 1]
    # plot and save figure
    plot_2D(xx, yy, names, figsize=figsize)
    plt.savefig(fig_path, bbox_inches='tight')
    

def plot_save_pca(embedding_vec, names, fig_path, figsize=(8, 8)):
    """ 
    embedding_vec: numpy array, shape (orig_dim, em_dim)
    names: list of str, len = orig_dim
    """
    pca = PCA(n_components=2)
    Y = pca.fit_transform(embedding_vec)
    xx, yy = Y[:, 0], Y[:, 1]
    # plot and save figure
    plot_2D(xx, yy, names, figsize=figsize)
    plt.savefig(fig_path, bbox_inches='tight')

    
# plot german states
fname = 'State'
print(les_dict[fname].classes_)
names = ['Berlin', 'Baden Wuerttemberg', 'Bayern', 'Niedersachsen/Bremen', 
         'Hessen', 'Hamburg', 'Nordrhein Westfalen', 'Rheinland Pfalz',
         'Schleswig Holstein', 'Sachsen', 'Sachsen Anhalt', 'Thueringen']
fig_path = os.path.join(image_dir, 'state_embedding.pdf')
plot_save_pca(embeddings_dict[fname], names, fig_path, figsize=(6, 7))
#plot_save_tsne(embeddings_dict[fname], names, fig_path, figsize=(6, 7))

#plot day of week
fname = 'DayOfWeek'
print(les_dict[fname].classes_)
names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
fig_path = os.path.join(image_dir, 'dow_embedding.pdf')
plot_save_pca(embeddings_dict[fname], names, fig_path, figsize=(4, 4))

"""
# plot day of week, (sin, cos) encoding 
N = 7
n = np.arange(N)
xx = np.sin(2 * np.pi * n / N)
yy = np.cos(2 * np.pi * n / N)
plot_2D(xx, yy, names, figsize=(4, 4))
fig_path = os.path.join(image_dir, 'dow_sin_cos.pdf')
plt.savefig(fig_path, bbox_inches='tight')
"""

#plot month
fname = 'Month'
print(les_dict[fname].classes_)
names = ['Jan', 'Oct', 'Nov', 'Dec', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
fig_path = os.path.join(image_dir, 'month_embedding.pdf')
plot_save_pca(embeddings_dict[fname], names, fig_path, figsize=(6, 6))


#plot day of month
fname = 'Day'
print(les_dict[fname].classes_)
names = list(les_dict[fname].classes_)
fig_path = os.path.join(image_dir, 'day_embedding.pdf')
plot_save_pca(embeddings_dict[fname], names, fig_path, figsize=(6, 6))


"""
plot pair-wise distances of stores
"""
#distances = np.loadtxt(open("distances.csv","r"), delimiter=" ")
#plt.figure(figsize=(8, 6))
#plt.scatter(distances[:, 0], distances[:, 1], edgecolor='k')
#plt.xlim([0, 20000])
#plt.xlabel('distance in metric space')
#plt.ylabel('distance in embedding space')
#plt.savefig(image_dir+'distance.pdf', bbox_inches='tight')
