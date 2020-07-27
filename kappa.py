class CohenKappa(tf.keras.metrics.Metric):
    """Cohen quadratic weighted kappa implementation for Keras."""
    
    def __init__(self, 
                 num_classes, 
                 sparse_input=True,  # true values
                 normalized_input=False, # predicted values
                 **kwargs):
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self._conf_mat_shape = tf.TensorShape((num_classes, num_classes))
        
        self.sparse_input = sparse_input
        if sparse_input:
            self._to_categorical = tf.keras.utils.to_categorical
            
        self.normalized_input = normalized_input
        if not normalized_input:
            self._normalize = tf.linalg.normalize

        self.count = self.add_weight(name="count",
                                     initializer="zeros")
        self.conf_mat = self.add_weight(name="confusion matrix",     
                                        shape=self._conf_mat_shape,
                                        initializer="zeros")
                                        
    def reset_states(self):
        """
        Overriding super().reset_states because it 
        does not appear to support non-scalar weights.
        """
        self.count.assign(0)
        self.conf_mat.assign(tf.zeros(self._conf_mat_shape))

    def update_state(self, true, pred, sample_weight=None):
        true = tf.convert_to_tensor(true)
        pred = tf.convert_to_tensor(pred)
        self._last_true = true
        if self.sparse_input:# and true.shape[0] is not None:
            true = tf.one_hot(tf.cast(true, tf.uint8), depth=self.num_classes)
            
        self._last_true = true
        
        if not self.normalized_input:
            pred = self._normalize(pred)[0]
        
        self._last_pred = pred
        
        assert pred.shape == true.shape
        
        self._last_count = pred.shape[0]
        if self._last_count is not None:
            self.count.assign_add(self._last_count)
        
        self.conf_mat.assign_add(
            tf.matmul(
                tf.reshape(tf.reduce_sum(pred, axis=0), (self.num_classes,1)),
                tf.reshape(tf.reduce_sum(true, axis=0), (1,self.num_classes))
            )
        )

    def result(self):
        return kappa(self.conf_mat, self.count, N=self.num_classes)

def kappa(conf_mat, count, N=5, weights=None):
    assert conf_mat.shape == (N, N)
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
        
    
    pred_total = tf.reduce_sum(conf_mat, axis=1, keepdims=True)
    true_total = tf.reduce_sum(conf_mat, axis=0, keepdims=True)
    
    pred_ratio = pred / count
    true_ratio = true / count
    expect = pred_ratio * true_ratio
        
    conf_mat_ratio = conf_mat / count
    
    return 1 - (tf.reduce_sum(weights * conf_mat_ratio) / \
                tf.reduce_sum(weights * expect) )
