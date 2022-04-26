# Machine Learning
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
* https://www.ics.uci.edu/~smyth/courses/cs274/readings/xing_singh_CMU_bias_variance.pdf

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

## L1 Vs L2 Regularisation
* L1 is also known as Lasso = sum of abolute of weights
* L2 is also known as Ridge = sum of square of weights

* L1 is used for Feature Selection as it tends to make weights to zero.
* Since L1 has a modulus of weights it is not differentiable in certain cases and hence other methods should be used for optimization.
* L2 is differentiable

https://www.youtube.com/watch?v=QNxNCgtWSaY&ab_channel=StatQuestwithJoshStarmer

* In case of L2, it doesnt make weights to Zero because, the zero error should be on the axis line for it to intersect with the locus of the L2, which means one of the weights is already zero. This in not likely and if the weights are already zero you do not need regularisation. In case of L1, this condition does not hold and the contour lines of error intersect on the axis.

## Decision Trees

* Creates Tree by Node Creation based upon the Algos like Information Gain and Gini Impurity.
* The Idea is creation of node should decrease the entropy.
* Gini(Impurity/Entropy at a node): 1 - (probability of Class of A )^2 - (probability of Class of B )^2
* Information Gain: Sigma P*log(P)

### Random Forests
* Decision Based on Max Votes from different decision Trees.
* Bagging is done to ensure decrease co-relation among each DT.
* Bagging randomly samples data with replacement.

### Bagging Vs Boosting
* Bagging decreases variance
* Boosting decreases Bias (Xtreme Gradient Boosting)

## SVM (Support Vector Machines )
* Source: Andrew NG
* In SVM we try to minimize the hinge loss function
#### Hinge loss: max(0, 1- y*(wT.x - b ))
* We have an additional parameter to be minimized i.e |w| = Sigma Theta^2 (sum of squares of weights)
* The above is because in SVM distance between the margin is 2/|w|. In order to maximize thi
* s we minimize |W|
#### Support Vectors
* Support Vectors play a major role in deciding the decision boundary
* For the Vectors are close to the decision boundary, the component of vecctor X, on the weight vector W, is small. In order to make it greater than 1, |W| should be greater. Since wT.X = |b|*|W| (W is not a UNIT weight vector)
* But, in the optimization function we specify to minimize the value of |w|. Hence the SVM chooses a Margin away from these Support Vectors.
#### Kernels
* Kernels help to acheive complex non -linear boundaries easily.
* Instead of having huge no of parameters like in logistic regression, the same  can be acheived through kernels.
* Few Kernels are, Gaussian Kernels, Polynomial Kernels (WT.X + constant)^degree, Linear Kernels(No Kernel)

## Optimisation functions
https://ruder.io/optimizing-gradient-descent/
#### SGD, Momentum, Nestrov, ADAM etc.

## Fine Tuning Vs Feature Extarction
https://i.stack.imgur.com/BiClc.png

## FewShot Learning
https://research.aimultiple.com/few-shot-learning/
https://www.borealisai.com/en/blog/tutorial-2-few-shot-learning-and-meta-learning-i/

## Activation Functions

Link: https://karpathy.medium.com/yes-you-should-understand-backprop-e2f06eab496b

Many Times in Using Activation functions and backprop we take granted that backprop happens automatically.
But this isnt the case.
### Sigmoid
* In case of using a sigmoid activation function. If the weight matrix is inialized with large values (negative and postive extrememus), the gradient will be zero. Hence the network stops learning. 
* Also, the gradient becomes exactly half of the output, so as you backprop the gradients become half.
### RELU
* In case of RELU, if the output is less than or equal to zero, then the gradient is zero, no learning happening.

### RNNs 
* In RNNs while backprop a single matrix gets multiplied many times, hence the gradients either get explode or diminish based upon its value. So either Use LSTM or use gradient clipping for exploding gradients.

## Linear Discrimination functions for MultiClass
If we wwant to solve a K class classification problem, we can either use (K-1) ( ONE vs ALL ) classifiers or K*(K-1)/2 classifiers for each pair of classes, when we use these classifiers, a class is assigned based upon maximum voting. 

However these classifiers suffer from the problem of ambugity regions, where voting doesnot help solve the problem. So we will need a single connected convex regions for classification. 

  for each class k,    Yk(x) > Yj(x) for all j classes then assign the point to class k.
P.S: Refer to the images attached.

<img width="662" alt="Screenshot 2022-04-26 at 11 21 07 AM" src="https://user-images.githubusercontent.com/9864247/165232400-e50d69c5-a362-4f64-aeed-cf439e99476c.png">

<img width="647" alt="Screenshot 2022-04-26 at 11 21 19 AM" src="https://user-images.githubusercontent.com/9864247/165232612-d099c921-c4e5-4843-813d-3c4c9c42da66.png">

## Normalization:
Batch normalization: Normalize each input feature across a mini batch of samples
Layer normalization: Normalize each input feature across all the features
weight normalization: 

