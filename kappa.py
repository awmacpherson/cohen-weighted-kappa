import tensorflow as tf

class CohenKappa(tf.keras.metrics.Metric):
    """Cohen quadratic weighted kappa implementation for Keras."""
    
    def __init__(self, 
                 num_classes, 
                 sparse_input=True,  # true values
                 weights='quadratic',
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.weights = weights
        self._shape = tf.TensorShape((num_classes, num_classes))
        self.sparse_input = sparse_input

        self.confusion = self.add_weight(name="confusion matrix",     
                                         shape=self._shape,
                                         initializer="zeros")
                                        
    def reset_states(self):
        """
        Overriding super().reset_states because it 
        does not support non-scalar weights.
        """
        self.confusion.assign(tf.zeros(self._shape))

    def update_state(self, true, pred, sample_weight=None):
        if self.sparse_input:
            true = one_hot(true, depth=self.num_classes)
          
        pred = discretize(pred)
        
        self.confusion.assign_add(confusion(true, pred)

    def result(self):
        return kappa(self.confusion, weights=self.weights)

def kappa(confusion, weights=None):
    N = tf.shape(confusion)[0]
    # Allow users to pass a predefined weight matrix.
    if weights == 'quadratic':
        idx = tf.range(N, dtype='float32')[:, None]
        row_idx = tf.tile(idx, (1,N))
        col_idx = tf.transpose(row_idx)
        weights = (row_idx - col_idx)**2 / ((N - 1) ** 2)
    elif weights is None:
        weights = tf.ones((N, N)) - tf.eye(N)
    elif not hasattr(weights, 'shape'):
        raise TypeError("weights must be either an array or 'quadratic'.")
    elif weights.shape != (N, N):
        raise ValueError("Weight matrix must be square with N rows.")
        
    pred = tf.reduce_sum(confusion, axis=1, keepdims=True)
    true = tf.reduce_sum(confusion, axis=0, keepdims=True)
    count = tf.reduce_sum(confusion)
    expected = pred_ratio * true / count
    
    return 1 - (tf.reduce_sum(weights * confusion) / \
                tf.reduce_sum(weights * expected) )

# Utility functions

def one_hot(z, depth):
    """One-hot encoding for a batch of scalars with dtype float32."""
    return tf.one_hot(tf.cast(z, tf.uint8), depth=depth)[:,0,:]

def discretize(z):
    z_class = tf.argmax(z, axis=-1)
    return tf.one_hot(z_class, depth=tf.shape(z)[-1])

def confusion(y_1, y_2):
    return tf.matmul( tf.transpose(tf.reduce_sum(y_1, axis=0, keepdims=True)),
                      tf.reduce_sum(y_2, axis=0, keepdims=True) )
