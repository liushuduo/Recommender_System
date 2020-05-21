import numpy as np
from scipy.stats import pearsonr

from copy import deepcopy
import warnings
import os

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


def load_data(path, m, n):
    """
    Load rating data in the specific format: 1 1 1\n1 2 1...
    :param path: path of file
    :param m: number of rows, i.e., the number of users
    :param n: number of columns, i.e., the number of items
    :return: user-item rating matrix in numpy array
    """
    # initialize dataset with zeros
    dataset = np.zeros([m, n])
    with open(path, 'r') as f:
        for line in f:
            idx = list(map(int, line.split()))
            dataset[idx[0] - 1, idx[1] - 1] = idx[2]  # entry 1 and 2 are index, entry 3 is vote.
    return dataset


def save_data(dataset, filename):
    """
    Save the predicted user-item rating matrix to file
    :param dataset: dataset to be saved
    :param filename: the name of saved file
    :return: none
    """
    with open(filename, 'w', encoding='ascii') as f:
        for idx, data in np.ndenumerate(dataset):
            if sum(idx) == 0:
                f.write('%s %s %.3f' % (idx[0] + 1, idx[1] + 1, data))
            else:
                f.write('\n%s %s %.3f' % (idx[0] + 1, idx[1] + 1, data))
    logger.info('Dataset saved in %s' % (os.path.join(os.getcwd(), filename)))


def mean_rate(dataset):
    """
    Calculate the average rating of per user exclude 0 rating
    :param dataset: user-item rating matrix
    :return: average rating in numpy array
    """
    temp_dataset = dataset.copy()  # original dataset is not changed
    temp_dataset[temp_dataset == 0] = np.nan
    return np.nanmean(dataset, 1)


class collabFilter(object):
    """
    Implementation of collaborative filter.
    """
    def __init__(self, dataset_path, users_num, items_num, corr_threshold=0.3,
                 nnbors=20):
        """
        Initialize the collaborative filter
        :param dataset_path: path of train data
        :param users_num: number of users
        :param items_num: number of items
        :param corr_threshold: threshold for selecting neighborhood
        :param nnbors: the number of neighborhood to select
        """
        logger.info("Starting up the Recommender System...")
        self.users_num = users_num
        self.items_num = items_num
        self.corr_threshold = corr_threshold
        self.nnbors = nnbors
        # Load user-item rating matrix
        self.dataset = load_data(dataset_path, self.users_num, self.items_num)
        # average rating of per user
        self.mean_rate = mean_rate(self.dataset)
        # number of rated items
        self.rated_items = np.where(self.dataset != 0)[0].size
        self.prediction = deepcopy(self.dataset)
        # calculate similarity
        self.similarity = np.diag(0.5 * np.ones(self.users_num))
        logger.info("Starting to calculate similarity matrix...")
        self.__similarity_mat()
        logger.info("Similarity matrix calculation complete!")

    def __similarity_mat(self):
        """
        Calculate the similarity between users and store as matrix.
        matrix[i, j] denotes the correlation between user i and j.
        :return: none
        """
        for active_user in range(self.users_num):
            if active_user % 10 == 0:
                logger.info("Progress: %.2f%%" % (100*active_user/self.users_num))
            self.similarity[active_user] = self.__pearson_corr(active_user)

    def save_prediction(self, filename):
        """
        Save the prediction to file.
        :param filename: name of saved file
        :return: none
        """
        save_data(self.prediction, filename)

    def predict_all(self):
        """
        Make prediction of all unrated items. The prediction result is stored in self.prediction.
        Available prediction is constrained in [1, 5]
        :return: none
        """
        uncovered = 0
        logger.info("Starting to predict ratings...")
        for active_user in range(self.users_num):
            if active_user % 30 == 0:
                logger.info("Progress: %.2f%%" % (100 * (active_user / self.users_num)))
            rating = self.prediction[active_user]
            # select items to predict
            predict_items = np.where(rating == 0)[0]
            for predict_item in predict_items:
                predict = self.predict_user_item(active_user, predict_item)
                if np.isnan(predict):
                    uncovered += 1
                rating[predict_item] = predict
        logger.info("All unrated items have been predicted! Coverage: %.2f%%" %
                    (100 * (1 - uncovered/(self.users_num*self.items_num))))

    def predict_user(self, active_user):
        """
        Make prediction for active user. The prediction is returned in numpy array.
        :param active_user: active user
        :return: predicted ratings in numpy array
        """
        ratings = deepcopy(self.dataset[active_user])
        predict_items = np.where(ratings == 0)[0]
        for predict_item in predict_items:
            ratings[predict_item] = self.predict_user_item(active_user, predict_item)
        return ratings

    def model_evaluation(self, ratio=0.1):
        """
        Evaluate the collaborative filter based on random selection of the training data.
        The mean absolute error (MAE) is utilized for evaluation.
        :param ratio: the evaluation dataset ratio
        :return: none
        """
        uncovered = 0
        user_MAE = np.array([])
        logger.info("Starting to evaluate model...")
        for idx, active_user in enumerate(
                np.random.choice(self.users_num, round(self.users_num * ratio), replace=False)):
            # save a copy for user's rating
            user_copy = self.dataset[active_user].copy()
            # find all rated items
            active_items = np.where(self.dataset[active_user] != 0)[0]
            # the unavailable prediction to be deleted are stored in del_item_ids
            del_item_ids = np.array([], dtype=np.int)
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
            user_MAE = np.append(user_MAE, np.mean(np.abs(predict_vec - user_copy[rating_label])))
            logger.info("MAE of user %s: %s. Uncovered items: %s. Progressing: %.2f%%" %
                        (active_user, user_MAE[-1], uncovered, 100 * (idx / round(self.users_num * ratio))))
        logger.info("Evaluation complete! MAE: %s. Coverage: %.2f%%" %
                    (np.mean(user_MAE), (100 * (self.rated_items - uncovered) / self.rated_items)))

    def predict_user_item(self, active_user, predict_item):
        """
        Predict the rating of predicted item for active user. If the prediction is not available,
        it will be assigned zero. Prediction is constrained from 1 to 5.
        :param active_user: active user
        :param predict_item: item to predict
        :return: prediction
        """
        # Calculate correlation/similarity between active user and the others
        corr = self.similarity[active_user]

        # Find neighborhood
        neighbors = self.__neighbor_select(corr, predict_item)
        if neighbors.size == 0:
            # if no neighbor user is available, rating cannot be predicted
            return np.nan
        # Make prediction
        predict = sum((self.dataset[neighbors, predict_item] -
                       self.mean_rate[neighbors]) * corr[neighbors]) \
                  / sum(corr[neighbors]) + self.mean_rate[active_user]
        # Constrain prediction value
        predict = np.clip(predict, 1, 5)
        return predict

    def __neighbor_select(self, corr, predict_item):
        """
        Find neighborhood of predicted item given the correlation between active user and the
        others. All users with correlation larger than self.corr_threshold will be considered as
        neighborhood of active user. If the available neighbors for current predicted item is
        larger than nnbors, at least nnbors neighbors will be selected.
        :param corr: correlation between active user and the others
        :param predict_item: item to predict
        :return: index of neighbor users in numpy array
        """
        available_user = np.where(self.dataset[:, predict_item] != 0)[0]
        if len(available_user) <= self.nnbors:
            # if available users is less than nnbors, the available users are used directly
            return available_user
        else:
            # else we choose satisfying users from available users
            abs_corr = np.abs(corr)
            abs_corr[available_user] += 1  # to make sure that available users are choose
        # the neighborhood with larger users are choose
        neighbors = np.where(abs_corr >= (self.corr_threshold + 1))[0]
        if len(neighbors) < self.nnbors:
            neighbors = np.argpartition(abs_corr, -self.nnbors)[-self.nnbors:]
        return neighbors

    def __pearson_corr(self, active_user):
        """
        Calculate the Pearson correlation between the active user and all the other users.
        Pearson correlation that cannot be calculated is assigned zero.
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
