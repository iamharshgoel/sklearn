# Machine Learning 
# Date:- 6th June, 2020
# Author:- Harsh Goel

Machine Learning falls into two main categories: supervised learning and unsupervised learning.

Supervised Learning

Supervised learning is the process of modelling the relationship between features of a dataset and targets (labels) associated with each sample of the dataset. With a model in hand, it is possible to use the model to either assign labels to a new dataset that doesn't yet have labels or calculate output values. The most common examples of supervised learning include: classification and regression.

Classification allows you to assign discrete **labels or categories** to new input data.
    Inputs                         Classification
    Texts, emails, or comments    Spam detection
    Flowers, insects, or animals  Species detection
    Viewers, readers, buyers      Customer detection

Regression analysis allows you to predict **continuous quantities** based on new input data.
    Inputs                                           Outputs
    
    Auto characteristics (color, model, age, etc)    Price
    Advertising dollars spent                        Sales revenue
    Candidate characteristics                        Salary

Unsupervised Learning

Unsupervised learning is the process of modeling relationships amongst features of a dataset in a way that classifies the raw data without supplying any input labels. There are many algorithms that enable relationships to be identified and each of these models seek to replicate human logic in finding patterns in data. Two of the most common unsupervised learning approaches are **clustering** and **dimensionality reduction**.

Cluster analysis or clustering is a technique for grouping a collection of objects so that all the objects in a single cluster are more similar to each other than to objects in other clusters.
    Inputs                   Classification
    
    Images                  Grouping/categorization
    Marketing data         Customer segmentation
    Social network data     Community classification

Dimensionality reduction (also dimension reduction) is the process of reducing the number of random variables in a dataset by identifying a set of principal variables. Dimensionality reductioncan be used for feature selection or feature extraction.
As an example, presume you have a dataset with 10 features for coffees:
* Cup size
* Roast (dark, etc)
* Flavoring (nutmeg, vanilla, etc)
* Country of origin
* Organic status (organic, not organic)
* Sustainability status (sustainably harvested?)
* Preparation (espresso, latte, etc)

If, through dimensionality reduction, we can determine that the most influential determinant of whether a coffee will sell well is cup size, roast, flavoring, and preparation, we may be able to speed up our analysis OR reduce our computational overhead by reducing the 10 features down to three OR four.

In some cases, data analysis such as regression or classification can be done in the reduced space more easily and/or accurately than in the original space.

Some benefits from using dimensionality reduction include:
* It reduces the computation time and storage space requirements.
* It can enable easier data visualization if the dimensions can be reduced to much lower dimensions like 2D/3D.
* It can improve the interpretation of the parameters of a machine learning model.
* It helps to avoid issues related to increase in data sparsity as data volume increases.  

Key Characteristics of Scikit-Learn:-

Scikit-Learn is a well known package that provides access to many common machine learning algorithms through a consistent, well-organized Application Programming Interface (API) and is supported by very thorough and comprehensive documentation.

The uniform syntax and the consistency in how the API is designed means that once you learn one model, it is surprisingly easy to pick up additional models.

A key goal of this repository is:-

* understanding the API\n",
* with an improved knowledge of the vocabulary of machine learning\n",
* knowing how to learn more


By and large, using any given model in Scikit-Learn will follow a set of straightforward steps. Each of our examples will follow what I call The Process:-

1. Prep the data:- the data must be well prepared for it to be usable in the various models. This preparation may include normalization, cleansing, wrangling of the data. It often needs to be separated into a `features` matrix and a `target` vector (array) and/or may need to be broken into separate collections of data for training versus testing purposes.\n",

2. Choose the model:- to choose a model, we will import the appropriate estimator class.
   
3. Choose appropriate hyperparameters:- to prepare the model, we create a class instance and provide hyperparameters as arguments to the class.
 
4. Fit the model:- to fit the model to the existing data, we call the `.fit()` method on the model instance and provide training data.

5. Apply the model:- next, we apply the model to new data, primarily by calling one of two methods:-

* Supervised learning: generally, we use the `.predict()` method to predict new labels.

* Unsupervised learning: generally, we use either the `.predict()` or `.transform()` methods to predict properties OR transform properties of the data.

6. Examine the results:- lastly, it is recommended that we look over the results and do a sanity check. Some of this can be done by simply looking at output values. Other times it really helps to have some form of data visualization (i.e. graph/chart) to help us examine the model predictions or transformations.


Data Handling

Generally, scikit-learn uses several of the most popular datatypes found in the Python data science ecosystem:

1. numpy arrays
2. scipy sparse matrixes
3. pandas DataFrames

To train a scikit-learn classifier, all you need is a 2D array (often called a features matrix and typically labeled X) for the input variables and a 1D array (often called a target array and typically labeled y) for the target labels.

Numpy
To see just the first few rows of a nump array, it is common to take a slice.

Pandas
To see just the first few rows of a pandas Series or DataFrame, it is common to use .head() or to take a slice.

Using sklearn.model_selection.train_test_split

An important component of machine learning is testing the models for some level of accuracy. It is customary to break the data into two OR more portions:

Set	        Purpose
Training	Used to train the model
Validation	Used to validate the model
Test	        Used to test the results of the validated training


Linear Regression

Linear Regression models are popular machine learning models because they:

* are often fast
* are often simple with few tunable hyperparameters
* are very easy to interpret
* can provide a nice baseline classification to start with before considering more sophisticated models

Several cases where you might use a linear regression to predict an output based on a set of inputs include:

Inputs	                Outputs
ad dollars spent	sales dollars earned
car age	                sale price
latitude	        skin cancer mortality

The LinearRegression model that we will examine here relies upon the Ordinary Least Squares (OLS) method to calculate a linear function that fits the input data.
Geometrically, this is seen as the sum of the squared distances,between each data point in the set and the corresponding point on the regression surface – the smaller the differences, the better the model fits the data.

Scikit-Learn has a number of Linear Models based on calculations besides OLS:

* Ridge
* Lasso
* Huber
and many more...

Each one has slightly different approaches to calculating a line that fits the data.

1. Ridge: addresses some issues related to OLS by controlling the size of coefficients.

2. Lasso: encourages simple, sparse models (i.e. models with fewer parameters). Can be useful when you want to automate certain parts of model selection, like variable selection/parameter elimination.

3. Huber: applies a linear loss (lower weight) to samples that are classified as outliers, thus minimizing the impact of random outliers.

Naive Bayes (GaussianNB)

Gaussian Naive Bayes has a method you can call that allows you to update models and can be used if the dataset is too large to fit into memory all at once using partial_fit() method.
Document classification and spam filtering are the real-world examples of GaussianNB.

Naive Bayes Classification models are popular machine learning models because they:

* are fast
* are simple with few tunable hyperparameters
* are suitable for datasets with very high dimensions
* can provide a nice baseline classification to start with before considering more sophisticated models

Naive Bayes Classifiers rely upon Bayes Theorem that allows you to predict the probability of a label if given some set of features:
P(label | features)

Scikit-learn has a number of Naive Bayes Classifiers. They are referred to as naive because they make certain presumptions about the data.

Each of the following has slightly different assumptions about the data. For example, the GaussianNB model that we will look at presumes that the "likelihood of the features is assumed to be Gaussian" (i.e. the likelihood of any given feature falls on a bell curve).

* BernoulliNB
* ComplementNB
* GaussianNB
* MultinomialNB

k-Means Clustering

The goal of a clustering algorithm is to assign data points to the same group if they are similar and to assign data points to different groups if they are different.

Clustering models are popular machine learning models because they:

* are unsupervised and thus don't require pre-determined labels
* can accommodate multidimensional datasets
* can, for simple cases, be fairly easy to interpret, especially in 2D/3D via charts

The k-Means Clustering algorithm:

* looks for the arithmetic mean of all points in a cluster to identify the cluster centers
* groups points together by identifying the closest cluster center

For this example, we will use the KMeans model. The sklearn.cluster module has a number of clustering models, including:

* AffinityPropagation
* DBSCAN
* KMeans
* MeanShift
* SpectralClustering
and more...


The k-Means Clustering model works based on a process called Expectation-Maximization. In this process, the model:

* starts by randomly picking some cluster centers
* repeats the following cycle until the model converges

Expectation: assign points to the closest cluster center
Maximization: use the points of the newly formed clusters to calculate a new mean to use as a new cluster center

The process is designed such that for every cycle of the Expectation and Maximization steps, the model will always have a better estimation of any given cluster.

Remember, in scatter plots:

* c values are assigned based on the labels we provide
* cmap maps a color to each value associated with c
* seismic is a range of colors from deep blue to deep red

PolynomialFeatures

The PolynomialFeature class has a .fit_transform() method that transforms input values into a series of output values. These values are often used as inputs in other models.

PolynomialFeatures generates a new feature matrix that has all the polynomial combinations of the original features with a degree less than or equal to the specified degree.

As an example:

An input sample has two dimensions (i.e. [a, b]) the resulting degree-2 polynomial features will be [1, a, b, a^2, ab, b^2].

Pipelines

In some cases, it might be necessary to transform the data in some way before feeding it into a particular machine learning model.

The data may need to be:

* scaled (ie. using StandardScaler
* changed into another format (ie. using PolynomialFeatures or CountVectorizer)
* normalized (i.e. using TfidfTransformer)

Pipelines allow you to feed inputs into one "end" of a series of components and get transformations or predictions out the other end, without having to take the output of one model and manually drop it into the inputs of the next model.

The following example uses the PolynomialFeatures model to transform inputs from a degree 1 polynomial into higher degree polynomials. It then takes the results of those transformations and then feeds them into the LinearRegression model.

The Pipeline simplifies things so that we only have to call .fit() once on the pipeline.
