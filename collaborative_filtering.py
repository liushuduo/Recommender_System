import numpy as np
from scipy.stats import pearsonr
from copy import deepcopy

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
        self.users_num = users_num
        self.items_num = items_num
        self.corr_threshold = corr_threshold
        self.nnbors = nnbors
        # Load user-item rating matrix
        self.dataset = load_data(dataset_path, self.users_num, self.items_num)
        self.mean_rate = mean_rate(self.dataset)  # average rating of per user

    def model_evaluation(self):
        uncovered = 0
        user_MAE = np.array([])
        logger.info("Starting to evaluate model...")
        for active_user in range(self.users_num):
            logger.info("Evaluating user %s...", active_user)
            # save a copy for user's rating
            user_copy = self.dataset[active_user].copy()
            # find all rated items
            active_items = np.where(self.dataset[active_user] != 0)[0]

            del_item_ids = np.array([])
            predict_vec = np.array([])
            for item_id, active_item in enumerate(active_items):
                # predict rating of active item given all the other ratings
                self.dataset[0, active_item] = 0
                predict = self.predict_user_item(0, active_item)
                if np.isnan(predict):
                    # current item cannot be predicted
                    uncovered += 1
                    # the item is deleted to calculate MAE
                    del_item_ids = np.append(del_item_ids, item_id)
                else:
                    # if prediction available
                    predict_vec = np.append(predict_vec, predict)
                # restore the original data
                self.dataset[0, active_item] = user_copy[active_item]
            rating_label = np.delete(active_items, del_item_ids)
            print(user_copy[rating_label])
            user_MAE = np.append(user_MAE, np.mean(np.abs(predict_vec - user_copy[rating_label])))
            print(user_MAE[-1])
        print("MAE: %s", np.mean(user_MAE))

    def predict_user(self, active_user):
        ratings = deepcopy(self.dataset[active_user])
        predict_items = np.where(ratings == 0)[0]
        for predict_item in predict_items:
            print(predict_item)
            ratings[predict_item] = self.predict_user_item(active_user, predict_item)
        return ratings

    def predict_user_item(self, active_user, predict_item):
        # Calculate correlation/similarity between active user and the others
        corr = self.__pearson_corr(active_user)

        # Find neighborhood
        neighbors = self.__neighbor_select(corr, predict_item)
        if neighbors.size == 0:
            # if no neighbor user is available, rating cannot be predicted
            return np.nan
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
    recommender = collabFilter("./train.txt", 943, 1682)
    recommender.model_evaluation()


