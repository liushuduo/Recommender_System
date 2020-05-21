from collaborative_filtering import collabFilter


def main():
    # create recommender system, 943 users and 1682 items
    recommender = collabFilter("train.txt", 943, 1682)
    recommender.predict_all()
    recommender.save_prediction("submit_result.txt")
    recommender.model_evaluation()

if __name__ == '__main__':
    main()
