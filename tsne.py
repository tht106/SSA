import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
import numpy as np
import torch
import random

from feeder.ntu_feeder import Feeder_single


ntu60_val_data = np.load('./data/gty/action_dataset/ntu60_frame50/xsub/val_position.npy')
pku_val_data = np.load('./data/gty/action_dataset/pku_part1_frame50/xsub/val_position.npy')

ntu60_val_data = random.choices(ntu60_val_data, k=1000)
pku_val_data = random.choices(pku_val_data, k=1000)

stacked_frames_ntu = np.stack(ntu60_val_data).reshape(1000 * 50, -1)
stacked_frames_pku = np.stack(pku_val_data).reshape(1000 * 50, -1)


tsne = TSNE(n_components=2, init='pca', random_state=0)
tSNE_result_training = tsne.fit_transform(stacked_frames_ntu)
tSNE_result_testing = tsne.fit_transform(stacked_frames_pku)

fig = plt.figure()
plt.scatter(tSNE_result_training[:,0], tSNE_result_training[:,1], c='red', s=1.0 )
plt.scatter(tSNE_result_testing[:,0], tSNE_result_testing[:,1], c='blue', s=1.0 )

plt.axis("tight")
plt.show()