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
