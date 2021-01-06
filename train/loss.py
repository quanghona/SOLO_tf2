import tensorflow as tf


class SOLOLoss(tf.keras.losses.Loss):
    """Loss for SOLO network

    Usage: This class can be use for built-in training method (keras) or custom
    training procedure
    - For built-in method, get the loss functions by invoke `get_category_loss`
    and `get_mask_loss` method and loss.weights to get the weights
    - For custom training function, Just invoke the functional method to get all
    losses at the same time

    Note: if call method is apply instead of __call__, please add `reduction`
    parameter to the constructor
    """
    def __init__(self, mask_weight=3, d_mask='dice', focal_gamma=2.0, focal_alpha=0.25, name='solo_loss'):
        super(SOLOLoss, self).__init__(name=name)
        self.mask_weight = mask_weight
        self.weights = [1, mask_weight]
        self.d_mask = d_mask
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha
        self.category_loss = FocalLoss(focal_gamma, focal_alpha)
        self.mask_loss = SOLOMaskLoss(d_mask, focal_gamma, focal_alpha)

    # def call(self, y_true, y_pred):
    #     l_cate = self.category_loss(y_true[0], y_pred[0])
    #     l_mask = self.mask_loss(y_true[1], y_pred[1])
    #     total_loss = l_cate + self.mask_weight * l_mask
    #     return total_loss   # We can only output total_loss as result if we override call method, but can apply tf.keras.losses.Reduction

    def __call__(self, y_true, y_pred):
        l_cate = self.category_loss(y_true[0], y_pred[0])
        l_mask = self.mask_loss(y_true[1], y_pred[1])
        total_loss = l_cate + self.mask_weight * l_mask
        return total_loss, l_cate, l_mask   # output 3 losses for tracking

    def get_catergory_loss(self):
        return self.category_loss

    def get_mask_loss(self):
        return self.mask_loss

    def get_config(self):
        config = super(SOLOLoss, self).get_config()
        config['mask_weight'] = self.mask_weight
        config['d_mask'] = self.d_mask
        config['focal_gamma'] = self.focal_gamma
        config['focal_alpha'] = self.focal_alpha
        return config


class SOLOMaskLoss(tf.keras.losses.Loss):
    """
    Mask loss for SOLO network
    """
    def __init__(self, d_mask='dice', focal_gamma=2.0, focal_alpha=0.25, reduction=tf.keras.losses.Reduction.AUTO, name='solo_mask_loss'):
        """Create loss mask loss instance

        Args:
            d_mask (str, optional): one of 'dice', 'focal' or 'bce'. Defaults to 'dice'.
            focal_gamma (float, optional): gamma factor for focal loss, only apply when d_mask is 'focal'. Defaults to 2.0.
            focal_alpha (float, optional): alpha factor for folca loss, only apply when d_mask is 'focal'. Defaults to 0.25.
            reduction (tf.keras.losses.Reduction, optional): reduction type. Defaults to tf.keras.losses.Reduction.AUTO.
            name (str, optional): ops name. Defaults to 'solo_mask_loss'.

        Raises:
            ValueError: if d_mask is not one of the above
        """
        super(SOLOMaskLoss, self).__init__(reduction=reduction, name=name)
        self.d_mask = d_mask
        self.focal_gamma = focal_gamma
        self.focal_alpha = focal_alpha

        if d_mask == 'dice':
            self.mask_loss = DiceLoss()
        elif d_mask == 'focal':
            self.mask_loss = FocalLoss(focal_gamma, focal_alpha, reduction=tf.keras.losses.Reduction.NONE)
        elif d_mask == 'bce':
            self.mask_loss = tf.keras.losses.BinaryCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
        else:
            raise ValueError('Mask distance type not supported')

    def call(self, y_true, y_pred):
        d_mask = self.mask_loss(y_true, y_pred)                                     # shape (B, x, x, S^2)
        d_mask = tf.math.reduce_mean(d_mask, axis=[1,2])                            # shape (B, S^2)
        indicator = tf.cast(tf.math.reduce_sum(y_true, axis=[1,2]) > 0, tf.float32)
        n_pos = tf.math.reduce_sum(indicator, axis=1)
        n_pos = tf.math.maximum(n_pos, tf.ones(n_pos.shape, dtype=tf.float32))      # shape (B,), prevent divided by 0
        loss = tf.math.reduce_sum(indicator * d_mask, axis=1) / n_pos               # shape (B,)
        return loss

    def get_config(self):
        config = super(SOLOMaskLoss, self).get_config()
        config['d_mask'] = self.d_mask
        config['focal_gamma'] = self.focal_gamma
        config['focal_alpha'] = self.focal_alpha
        return config


class FocalLoss(tf.keras.losses.Loss):
    """
    Paper: https://arxiv.org/pdf/1708.02002.pdf

    Formula: FL(pt) = -alpha*(1 - pt)^gamma * log(pt)
    log(pt) = y_true * log(y_pred)
    """
    def __init__(self, gamma=2.0, alpha=0.25, reduction=tf.keras.losses.Reduction.AUTO, name='focal_loss'):
        super(FocalLoss, self).__init__(reduction=reduction, name=name)
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = 1e-7

    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, self.epsilon, 1 - self.epsilon)
        loss = -self.alpha * tf.math.pow(1 - y_pred, self.gamma) * y_true * tf.math.log(y_pred)
        return loss

    def get_config(self):
        config = super(FocalLoss, self).get_config()
        config['gamma'] = self.gamma
        config['alpha'] = self.alpha
        config['epsilon'] = self.epsilon
        return config


class DiceLoss(tf.keras.losses.Loss):
    """
    Paper: https://arxiv.org/pdf/1606.04797.pdf

    Dice loss based on deice coefficient
    """
    def __init__(self, keepdims=True, reduction=tf.keras.losses.Reduction.NONE, name='dice_loss'):
        """Create dice loss instance

        Args:
            keepdims (bool, optional): if true, the loss result will keep the image dimension (with length 1). Defaults to True.
            reduction (tf.keras.losses.Reduction, optional): reduction type. Defaults to tf.keras.losses.Reduction.NONE.
            name (str, optional): ops name. Defaults to 'dice_loss'.
        """
        super(DiceLoss, self).__init__(reduction=reduction, name=name)
        self.keepdims = keepdims

    def call(self, y_true, y_pred):
        pq = tf.math.reduce_sum(tf.math.multiply(y_pred, y_true), axis=[1,2], keepdims=self.keepdims)
        p2 = tf.math.reduce_sum(tf.math.multiply(y_pred, y_pred), axis=[1,2], keepdims=self.keepdims)
        q2 = tf.math.reduce_sum(tf.math.multiply(y_true, y_true), axis=[1,2], keepdims=self.keepdims)
        return 1 - 2 * pq / (p2 + q2)   # shape (B, 1, 1, S^2) if keepdims else (B, S^2)

    def get_config(self):
        config = super(DiceLoss, self).get_config()
        config['keepdims'] = self.keepdims
        return config
