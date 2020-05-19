import numpy as np
from scipy.stats import pearsonr

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_mat(path, m, n):
    dataset = np.zeros([m, n])
    with open(path, 'r') as f:
        for line in f:
            idx = list(map(int, line.split()))
            dataset[idx[0] - 1][idx[1] - 1] = idx[2]  # entry 1 and 2 are index, entry 3 is vote.
    return dataset


def pearson(vec_a, vec_b):
    if len(vec_a) <= 1:
        # for vector dim less than 2, the pearson correlation is assumed as 0
        return 0
    return pearsonr(vec_a, vec_b)


class collabFilter(object):
    """
        Implementation of collaborative filter.
    """

    def __init__(self, dataset_path, users_num, items_num):
        logger.info("Starting up the Recommender System...")
        # Load rating data
        self.users_num = users_num
        self.items_num = items_num
        self.dataset = load_mat(dataset_path, self.users_num, self.items_num)

    def prediction(self, active_user):
        """
        Predict ratings of active user.
        """
        corr = self.__pearson_corr(active_user)
        print(corr)

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
                corr_vec[user] = 0
            else:
                corr_vec[user], _ = pearsonr(self.dataset[active_user][intersection],
                                             self.dataset[user][intersection])
        return corr_vec


if __name__ == '__main__':
    a = collabFilter("./train.txt", 943, 1682)
    a.prediction(1)
