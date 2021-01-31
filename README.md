# ML
ML topics read and understood from various sources (Books, research papers, articles) and written by me.  

## No Free lunch Theorem:
* This theorem states that no machine learning algorithm is favoured over the other considering all the tasks.
* The best way is to tune any  ml algo specific to your task. 
* One way of doing this is adding new functions to the hypothesis function or by prioriting few functions over the other. This can be done by regularisation.
     
     `Regularisation:  + (lamda)*W^*W`

* Higher lambda leads to lower weights

## Underfitting Vs Overfitting 
* Undefitting of models happens when there is not enough training data. 
* It leads to bias in the models.
* Overfitting happens when you model noise along with patterns in the data. It leads to high variance.
* Variance can be reduced by using the small weights (high lamba value in the regularisation)

## i.i.d

* While Sampling data for ML models it is assumed that all samples are drawn independently from an identical (same) underlying distribution. 
 
## VC Dimension

* It is used to determine the models capacity. It is the number of samples a model can shatter(set theory concept)

## Hyper Parameters
* The parameters that are not optimised generally  as part of the optimization function. You can learn another algorithm to find the optimal Hyper Parameters or use the Validation data to determine the Hyper Parameters. (20% of Training Data is Used for Validation generally)

## Bias - Variance
* We to approximate the underlying function using the training data we have. But the training data does not represent the entire set of training samples, due to this error perisists in the approximation function. The following are the characteristics of the approximation function which help us in modelling the error understanding the nature of functions.
Given a function f,
* Bias =  Expectation[f] - f
* Variance is simply the vraiance of f.
* Variance[x] = E[x*x] - (E[x])^2
* Bias and Varaince can be modelled as a part of Mean Square Error. So When you are minimising MSE you are also minimising the Bias and Varaince.

## Cross Entropy Loss
* Many authors use the term “cross-entropy” to identify speciﬁcally the negative log-likelihood of a Bernoulli or softmax distribution,but that is a misnomer.
*  Any loss consisting of a negative log-likelihood is a cross-entropy between the empirical distribution deﬁned by the training set and the probability distribution deﬁned by model. 
* For example, mean squared error is the cross-entropy between the empirical distribution and a Gaussian model.

## Properties of Maximum Likelihood
* The true distribution pdata must lie within the model family pmodel(·;θ). Otherwise, no estimator can recover pdata
* The true distribution pdata must correspond to exactly one value of θ. Otherwise, maximum likelihood can recover the correct pdata but will not be able to determine which value of θ was used by the data-generating process.
* That parametric mean squared error decreases as m increases, and for m large, the Cramér-Rao lower bound (Rao, 1945; Cramér, 1946) shows that no consistent estimator has a lower MSE than the maximum likelihood estimator.
* For these reasons (consistency and eﬃciency), maximum likelihood is often considered the preferred estimator to use for machine learning.
* When the numberof examples is small enough to yield overﬁtting behavior, regularization strategies such as weight decay may be used to obtain a biased version of maximum likelihoodthat has less variance when training data is limited.
