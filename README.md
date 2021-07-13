# Demos

In this repository, I'll accumulate some miscellaneous small projects I've done, all of which will have a nice visual demonstration.

---

## Regularisation for Linear Regression

This demo is something I created as a TA for CS 4780 (Intro ML) at Cornell, in the Spring of 2021. It shows the effect of different types of regularisation on gradient descent for simple linear regression models.  

The animations demonstrate the ground truth points with a bit of noise added to them. These points are generated from simple linear and polynomial ground truth functions.  

The moving line demonstrates how gradient descent affects the convergence of the predicted function, and the boxes below demonstrate the (normalized) weight assigned to each feature as gradient descent progresses. The weights for each run are initialized to zero for consistency. 

### 1. No Regularisation, L1, and L2

- In the first video, you can see the impact of regularisation vs none of it.  
- In the left-most plot, the predicted function seems too susceptible to the added noise and overfits :( to the data. The distribution of weight seems quite even across the spurious polynomial features.  
- With L1 regularisation, the predicted line fits the general trend much better. The distribution of weights is sparse, with most of it concentrated on the (salient) linear feature.  
- L2 regularisation also avoids too much overfitting, but the final distribution of weight is smoother than L1.  
![No reg, L1 reg, L2 reg](https://user-images.githubusercontent.com/39067298/125436237-a78a7b10-c69c-46e2-85bb-cecd649db8f7.mp4)

### 2. Different amounts of L1 regularisation

- By varying the scaling factor (here denoted "lambda") of the regularisation term, we see the effect of different amounts of regularisation. Too little, and it overfits. Too much, and it underfits, failing to capture the trend of the data points.
- Gradient descent is like Goldilocks... it's quite fussy!
![Different amounts of lambda scaling for L1](https://user-images.githubusercontent.com/39067298/125437708-6f235028-3d11-466f-ad80-06c2d032e5fd.mp4)

### 3. L-Infinity Regularisation

- A student in my section asked me about L-Infinity regularisation, and I hadn't really thought of it before. I decided to plot it.
- The effect is pretty much what you would expect... quite a balanced spread of weight across the features as the weight for no one feature can dominate too much. Looks somewhat like no regularisation in this plot.
![L-inf regularisation](https://user-images.githubusercontent.com/39067298/125437803-ccd3a891-c5a4-4fe1-a8df-7c5aa5e3897d.mp4)

### 4. Uh-oh, forgot to tune the learning rate
![Kaboom. Gradient Descent dead](https://user-images.githubusercontent.com/39067298/125437951-3e396c16-ae1c-41eb-910e-1441f2ad0de5.mp4)

---



