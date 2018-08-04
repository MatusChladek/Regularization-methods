"""
Pathwise coordinate-wise descent
"""
class Lasso(BaseEstimator, RegressorMixin):
  def __init__(self, alpha=1.0, max_iter=1000, fit_intercept=True):
    self.alpha = alpha #Penalization coefficient
    self.max_iter = max_iter # Number of iterations
    self.fit_intercept = fit_intercept # Intercept
    self.coef_ = None
    self.intercept_ = None
    self.path = None



    """
    Soft Thresholding defined above

    """
  def _soft_thresholding_operator(self, x, lambda_):
    if x > 0 and lambda_ < abs(x):
      return x - lambda_
    elif x < 0 and lambda_ < abs(x):
      return x + lambda_
    else:
      return 0


  def fit(self, X, y):
    """
    Checking value of fit_intercept and adds one vector to X matrix
    """
    if self.fit_intercept:
      X = np.column_stack((np.ones(len(X)),X))

    beta = np.zeros(X.shape[1])
    if self.fit_intercept:
      beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:]))/(X.shape[0])

    path = np.vstack((beta,np.zeros([self.max_iter,len(beta)])))

    """
    Main iteration
    """
    for iteration in range(self.max_iter):
      start = 1 if self.fit_intercept else 0
      for j in range(start, len(beta)):
        tmp_beta = deepcopy(beta)
        tmp_beta[j] = 0.0
        r_j = y - np.dot(X, tmp_beta)
        arg1 = np.dot(X[:, j], r_j)
        arg2 = self.alpha*X.shape[0]

        beta[j] = self._soft_thresholding_operator(arg1, arg2)/(X[:, j]**2).sum()
        path[iteration+1,] = beta

        if self.fit_intercept:
          beta[0] = np.sum(y - np.dot(X[:, 1:], beta[1:]))/(X.shape[0])
          path[iteration+1,] = beta

    if self.fit_intercept:
      self.intercept_ = beta[0]
      self.coef_ = beta[1:]
      self.path = path
    else:
      self.coef_ = beta
      self.path = path

    return self

  def predict(self, X):
    y = np.dot(X, self.coef_)
    if self.fit_intercept:
      y += self.intercept_*np.ones(len(y))
    return y
