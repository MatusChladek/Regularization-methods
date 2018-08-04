# Packages
import scipy
import scipy.linalg
from scipy import array, log, c_, r_, shape, pi, sqrt, zeros, mean, linspace, \
    exp, diff, sign, arange, average, interp, floor, iterable, argmin, std, \
    var, ones, dot, argmax
from scipy.linalg import solve, cho_solve, norm
import pylab

# Utility functions updates R as a cholesky factorization
# dot(R.T,R) = dot(X.T,X) R is current cholesky matrix to be updated,
# x is the column vector representing the variable to be added and X
# is the data matrix with the currently active variables other than x
#
def cholinsert(R, x, X):
    diag_k = dot(x.T,x)
    if R.shape == (0,0):
        R = array([[sqrt(diag_k)]])
    else:
        col_k = dot(x.T,X)

        R_k = solve(R,col_k)
        R_kk = sqrt(diag_k - dot(R_k.T,R_k))
        R = r_[c_[R,R_k],c_[zeros((1,R.shape[0])),R_kk]]

    return R

def lars1(X, y):
    # n = number of data points, p = number of predictors
    n,p = X.shape

    # mu = regressed version of y sice there are no predictors it is initially the
    # zero vector
    mu = zeros(n)

    # active set and inactive set - they should invariably be complements
    act_set = []
    inact_set = list(range(p))

    # current regression coefficients and correlation with residual
    beta = zeros((p+1,p))
    corr = zeros((p+1,p))

    # initial cholesky decomposition of the gram matrix
    # since the active set is empty this is the empty matrix
    R = zeros((0,0))

    # add the variables one at a time
    for k in range(p):

        # compute the current correlation
        c = dot(X.T, y - mu)

        # store the result
        corr[k,:] = c

        # choose the predictor with the maximum correlation and add it to the active
        # set
        jmax = inact_set[argmax(abs(c[inact_set]))]
        C = c[jmax]


        # add the most correlated predictor to the active set
        R = cholinsert(R,X[:,jmax],X[:,act_set])
        act_set.append(jmax)
        inact_set.remove(jmax)

        # get the signs of the correlations
        s = sign(c[act_set])
        s = s.reshape(len(s),1)

        # move in the direction of the least squares solution restricted to the active
        # set

        GA1 = solve(R,solve(R.T, s))
        AA = 1/sqrt(sum(GA1 * s))
        w = AA * GA1


        u = dot(X[:,act_set], w).reshape(-1)


        # if this is the last iteration i.e. all variables are in the
        # active set, then set the step toward the full least squares
        # solution
        if k == p:
            gamma = C / AA
        else:
            a = dot(X.T,u)
            a = a.reshape((len(a),))

            tmp = r_[(C - c[inact_set])/(AA - a[inact_set]),
                     (C + c[inact_set])/(AA + a[inact_set])]

            gamma = min(r_[tmp[tmp > 0], array([C/AA]).reshape(-1)])


        mu = mu + gamma * u

        if beta.shape[0] < k:
            beta = c_[beta, zeros((beta.shape[0],))]
        beta[k+1,act_set] = beta[k,act_set] + gamma*w.T.reshape(-1)

    return beta
