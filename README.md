# Recommender System

This is a python project for Data Mining class 2020 summer, which implements a recommender system via *user-based collaborative filtering*. You can refer to the [paper of Herlocker et al.](https://dl.acm.org/doi/pdf/10.1145/3130348.3130372) and the [paper of Breese et al.](https://arxiv.org/abs/1301.7363) for a comprehensive review on collaborative filtering. 

## Install

This project uses ***python 3.7.3*** and a virtual environment is recommended. The required packages are listed in the [requirements.txt](./requirements.txt). Your can use the following command to install them automatically. 

```bash
pip install -r requirements.txt
```

## Usage

The implementation of collaborative filter is encapsulated in a python class named *collabFilter*. The parameters required for instantiating the class are the path of data, number of users and number of items, respectively. Two optional parameters decide the neighbor users to select. The dataset to be loaded must be an ASCII text file where each row is a non-zero element of rating matrix with three entries: **the user ID number, the item ID number and the rating value**. 

 The similarity matrix is calculated while instantiating. Using *model_evaluation* method gives the mean absolute error (MAE) of the filter. Finally, *predict_all* method fills all the unrated entry with 1.000-5.000 or nan if prediction is unavailable. Using *save_prediction* method with a filename to save the prediction result in the format as the input.

Run [main.py](./main.py) to get an evaluation of filter and the prediction with name [submit_result.txt](./submit_result.txt).

## Details

User-based collaborative method can be separated into three steps.

1. Weight all users with respect to similarity with the active user. Many metric can be used such Pearson Correlation, Spearman Correlation, and Vector Similarity. In this project we use the **Pearson Correlation**. The Pearson correlation is defined as 

   <center><img src="http://latex.codecogs.com/gif.latex?w_{a, u}=\frac{\sum_{i=1}^{m}\left(r_{a, i}-\bar{r}_{a}\right) *\left(r_{u, i}-\bar{r}_{u}\right)}{\sqrt{\sum_{i=1}^m(r_{a,i}-\bar{r}_a)} * \sqrt{\sum_{i=1}^m(r_{u,i}-\bar{r}_u)}}." /></center>

2. Select a subset of users to use as a set of predictors. This project combines **weight thresholding** with **best-n neighbors**, which can provide available predictions as much as possible.

3. Normalize ratings and compute a prediction from a weighted combination of selected neighbors' ratings. In this project, the prediction is made using 

   <center><img src="http://latex.codecogs.com/gif.latex?p_{a, i}=\bar{r}_{a}+\frac{\sum_{u=1}^{n}\left(r_{u, i}-\bar{r}_{u}\right) * w_{a, u}}{\sum_{u=1}^{n} w_{a, u}}." /></center>

   



<!--
<img src="http://latex.codecogs.com/gif.latex?" />
-->