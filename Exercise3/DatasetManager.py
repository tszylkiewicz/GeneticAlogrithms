import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA


class DatasetManager:

    def __init__(self):
        self.data = []
        self.target = []
        self.size = 0
        self.dimension = 0

    def load_dataset(self, dataset_name="iris"):
        temp = datasets.load_wine()
        self.data = temp.data
        self.target = temp.target
        self.size = self.data.shape[0]
        self.dimension = self.data.shape[1]

    def plot_data(self):
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c='black',
           edgecolor='k', s=30)        
        ax.set_title("First three directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])

        plt.show()

    def plot_clustered_data(self, ind, clusters):
        temp = np.reshape(ind, (int(len(ind)/self.dimension), self.dimension))
        fig = plt.figure(1, figsize=(8, 6))
        ax = Axes3D(fig, elev=-150, azim=110)
        ax.scatter(self.data[:, 0], self.data[:, 1], self.data[:, 2], c=clusters,
           edgecolor='k', s=30)
        ax.scatter(temp[:, 0], temp[:, 1], temp[:, 2], c='black',
                   edgecolor='k', s=50)
        ax.set_title("First three directions")
        ax.set_xlabel("1st eigenvector")
        ax.w_xaxis.set_ticklabels([])
        ax.set_ylabel("2nd eigenvector")
        ax.w_yaxis.set_ticklabels([])
        ax.set_zlabel("3rd eigenvector")
        ax.w_zaxis.set_ticklabels([])

        plt.show()

    def group_data(self, assignments, groups):
        grouped = {}
        for i in range(groups):
            grouped.setdefault(i, [])

        for idx, val in enumerate(assignments):
            grouped[val].append(self.data[idx])

        return grouped

    def get_data_bounds(self):

        lower_bounds = np.zeros(self.dimension)
        upper_bounds = np.zeros(self.dimension)

        for i in range(self.dimension):
            lower_bounds[i] = np.min(self.data[:, i])
            upper_bounds[i] = np.max(self.data[:, i])

        return lower_bounds, upper_bounds
