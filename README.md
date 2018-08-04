# Regularization methods
## Shrinkage methods for regression models restriction

There are two reasons why we are often not satisfied with the least squares
estimates.

* The first is *prediction accuracy*: the least squares estimates often have
low bias but large variance. Prediction accuracy can sometimes be
improved by shrinking or setting some coefficients to zero. By doing
so we sacrifice a little bit of bias to reduce the variance of the predicted
values, and hence may improve the overall prediction accuracy.

* The second reason is *interpretation*. With a large number of predictors,
we often would like to determine a smaller subset that exhibit
the strongest effects. In order to get the “big picture,” we are willing
to sacrifice some of the small details.

Some of the most often used methods for model restriction include *Forward- and Backward-Stepwise Selection* or *Stagewise regression*. By retaining a subset of the predictors and discarding the rest, subset selection
produces a model that is interpretable and has possibly lower prediction
error than the full model. However, because it is a discrete process—
variables are either retained or discarded—it often exhibits high variance,
and so doesn’t reduce the prediction error of the full model. *Shrinkage*
*methods* (namely Ridge,Lasso and Elastic Net) are more continuous and don’t suffer as much from high
variability.

As a continuous shrinkage method, Ridge regression achieves its better prediction performance through a bias–variance trade-off. However, ridge regression cannot produce a parsimonious model, for it always keeps all the predictors in the model.

A promising technique called the Lasso was proposed by Tibshirani (1996). The lasso is a penalized least squares method imposing an L1-penalty on the regression coefficients. Owing to the nature of the L1-penalty, the lasso does both continuous shrinkage and automatic variable selection simultaneously. On the other hand introduction of L1 penalty causes no closed form solution in non-orthonormal case. This makes computation of Lasso estimates a quadratic programming problem.

## Lasso estimation algorithms
### The LARS Algorithm
​
At the first step it identifies the variable
most correlated with the response. Rather than fit this variable completely,
LAR moves the coefficient of this variable continuously toward its leasts quares
value (causing its correlation with the evolving residual to decrease
in absolute value). As soon as another variable “catches up” in terms of
correlation with the residual, the process is paused. The second variable
then joins the active set, and their coefficients are moved together in a way
that keeps their correlations tied and decreasing.This process is continued until all the variables are in the model, and ends at the full least-squares
fit.
​
##### Naive LARS
​
1. Standardize the predictors to have mean zero and unit norm. Start
with the residual <img src="https://latex.codecogs.com/gif.latex?$r=y-\bar{y},&space;\beta_1,\beta_2,...&space;,\beta_p&space;=&space;0$" title="$r=y-\bar{y}, \beta_1,\beta_2,... ,\beta_p = 0$" />
2. Find the predictor $x_j$ most correlated with $r$
3. Move $\beta_j$ from 0 towards its least-squares coefficient $<x_j,r>$ until some
other competitor $x_k$ has as much correlation with the current residual
as does $x_j$
4. Move $\beta_j$ and $\beta_k$ in the direction defined by their joint least squares
coefficient of the current residual on $(x_j,x_k)$, until some other competitor
$x_l$ has as much correlation with the current residual
5. Continue in this way until all $p$ predictors have been entered. After
$min(N − 1,p)$ steps, we arrive at the full least-squares solution.
​
Tibshirani,Hastie&Friedman (2009) showed that LAR is almost identical to lasso path and differ only when coefficient crosses zero value. It appears that just simple modification of the LAR algorithm  gives the entire
lasso path, which is also piecewise-linear.
​
##### Lasso modification
​
* 4a. If a non-zero coefficient hits zero, drop its variable from the active set
of variables and recompute the current joint least squares direction.
​
All of the steps above introduce an efficient way for computing lasso having the same order of computation as Cholesky or QR decomposition which are used for least squares fitting.

### Path-wise coordinate descent algorithm

An alternate approach to the LARS algorithm for computing the lasso
solution is simple coordinate descent. This idea was proposed by Fu (1998)
and Daubechies et al. (2004), and later studied and generalized by Friedman
et al. (2007). The idea is to fix the penalty
parameter $\lambda$ in the Lagrangian form and optimize successively over
each parameter, holding the other parameters fixed at their current values. This method is also called “one-at-atime”
coordinate-wise descent algorithm in the literature.

A key point here: coordinate descent works so well because minimization can be done quickly,
and the relevant equations can be updated as we cycle through the variables.That makes it faster than LARS algorithm especially in large problems.

By rearranging general Lasso Lagrangian form we can view problem as univariate lasso problem with explicit solution resulting in update

$$\hat\beta(\lambda )← S(\sum_{i=1}^{N}{x_{ij}(y_i-\hat y^{(j)})},\lambda)$$
Here $S(\hat\beta, \lambda) = sign(\hat\beta)(|\hat\beta|−\lambda)_+$ is so called soft-thresholding operator. The first argument to $S(·)$ is the simple least-squares coefficient
of the partial residual on the standardized variable $x_{ij}$ . Repeated iteration
of updating function above—cycling through each variable in turn until convergence—yields
the lasso estimate $\hat\beta(\lambda)$
