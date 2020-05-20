import numpy as np
from scipy.stats import pearsonr

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_data(path, m, n):
    dataset = np.zeros([m, n])
    with open(path, 'r') as f:
        for line in f:
            idx = list(map(int, line.split()))
            dataset[idx[0] - 1, idx[1] - 1] = idx[2]  # entry 1 and 2 are index, entry 3 is vote.
    return dataset


def mean_rate(dataset):
    temp_dataset = dataset.copy()  # original dataset is not changed
    temp_dataset[temp_dataset == 0] = np.nan
    return np.nanmean(dataset, 1)


class collabFilter(object):
    """
        Implementation of collaborative filter.
    """

    def __init__(self, dataset_path, users_num, items_num, corr_threshold=0.3,
                 nnbors=30):
        logger.info("Starting up the Recommender System...")
        # Load rating data
        self.users_num = users_num
        self.items_num = items_num
        self.corr_threshold = corr_threshold
        self.nnbors = nnbors
        self.dataset = load_data(dataset_path, self.users_num, self.items_num)  # user-item rating matrix
        self.mean_rate = mean_rate(self.dataset)  # average rating of per user

    def prediction(self, active_user, predict_item):
        # Calculate correlation/similarity between active user and the others
        corr = self.__pearson_corr(active_user)

        # Find neighborhood
        neighbors = self.__neighbor_select(corr, predict_item)

        # Make prediction
        predict = sum((self.dataset[neighbors, predict_item] -
                       self.mean_rate[neighbors]) * corr[neighbors]) \
                  / sum(corr[neighbors]) + self.mean_rate[active_user]
        return predict

    def __neighbor_select(self, corr, predict_item):
        available_user = np.where(self.dataset[:, predict_item] != 0)[0]
        if len(available_user) <= self.nnbors:
            # if available users is less than nnbors, the available users are used directly
            return available_user
        else:
            # else we choose satisfying users from available users
            abs_corr = np.abs(corr)
            abs_corr[available_user] += 1  # to make sure that available users are choose

        # the neighborhood with larger users are choose
        neighbors = np.where(abs_corr >= (self.corr_threshold+1))[0]
        if len(neighbors) < self.nnbors:
            neighbors = np.argpartition(abs_corr, -self.nnbors)[-self.nnbors:]
        return neighbors

    def __pearson_corr(self, active_user):
        """
        Calculate the Pearson correlation between the active user and all the other users.
        :param active_user: index of active user
        :return: Pearson correlation between active user and the others
        """
        active_item = np.where(self.dataset[active_user] != 0)
        corr_vec = np.zeros(self.users_num)
        for user in range(self.users_num):
            intersection = np.intersect1d(np.where(self.dataset[user] != 0), active_item)
            if len(intersection) <= 1:
                # when the number of intersection is less than 2, the Pearson correlation is undefined.
                corr_vec[user] = 0
            else:
                corr_vec[user], _ = pearsonr(self.dataset[active_user, intersection],
                                             self.dataset[user, intersection])
                # if one of the two vectors is constant, the Pearson correlation is undefined.
        corr_vec[np.isnan(corr_vec)] = 0
        corr_vec[active_user] = 0
        return corr_vec


if __name__ == '__main__':
    a = collabFilter("./train.txt", 943, 1682)
    print(a.prediction(1, 1))
