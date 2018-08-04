 import keras.backend as K

def AndrewsWave(yTrue,yPred):
  """
  a : float, optional
        The tuning constant for Andrew's Wave function.  The default value is
        1.339.
  """
  a = 1.339
  z = yTrue - yPred
  test = K.abs(z) >= a * np.pi
  return tf.where(test, a*(1 - tf.cos(z/a)),2*a*(z-z+1))
