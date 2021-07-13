# Demos

In this repository, I'll accumulate some miscellaneous small projects I've done, all of which will have a nice visual demonstration.

---

## Regularisation for Linear Regression

This demo is something I created as a TA for CS 4780 (Intro ML) at Cornell. It shows the effect of different types of regularisation on gradient descent for simple linear regression models.  
The animations demonstrate the ground truth points with a bit of noise added to them. These points are generated from simple linear and polynomial ground truth functions.  
The moving line demonstrates how gradient descent affects the convergence of the predicted function, and the boxes below demonstrate the (normalized) weight assigned to each feature as gradient descent progresses. The weights for each run are initialized to zero for consistency. 

### 1. No Regularisation, L1, and L2

In the first video, you can see the impact of regularisation vs none of it.  
In the left-most plot, the predicted function seems too susceptible to the added noise and overfits :( to the data. The distribution of weight seems quite even across the spurious polynomial features.  
With L1 regularisation, the predicted line fits the general trend much better. The distribution of weights is sparse, with most of it concentrated on the (salient) linear feature.  
L2 regularisation also avoids too much overfitting, but the final distribution of weight is smoother than L1.  
![No reg, L1 reg, L2 reg](./regularisation/animations/animation_3reg.mp4)
