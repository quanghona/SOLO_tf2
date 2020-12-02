import tensorflow as tf

"""
CoordConv implementation which translatre to tensorflow version 2.x
Paper and original code: https://arxiv.org/pdf/1807.03247.pdf
"""

'''
class AddCoords(tf.keras.layers.Layer):
    """Add coords to a tensor"""
    def __init__(self, x_dim=64, y_dim=64, with_r=False):
        super(AddCoords, self).__init__()
        self.x_dim = x_dim
        self.y_dim = y_dim
        self.with_r = with_r

    def call(self, input_tensor):
        """
        input_tensor: (batch, x_dim, y_dim, c)
        """
        batch_size_tensor = tf.shape(input_tensor)[0]
        xx_ones = tf.ones([batch_size_tensor, self.x_dim], dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(self.y_dim), 0), [batch_size_tensor, 1])
        xx_range = tf.expand_dims(xx_range, 1)

        xx_channel = tf.matmul(xx_ones, xx_range)
        xx_channel = tf.expand_dims(xx_channel, -1)

        yy_ones = tf.ones([batch_size_tensor, self.y_dim], dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(self.x_dim), 0), [batch_size_tensor, 1])
        yy_range = tf.expand_dims(yy_range, -1)

        yy_channel = tf.matmul(yy_range, yy_ones)
        yy_channel = tf.expand_dims(yy_channel, -1)

        xx_channel = tf.cast(xx_channel, tf.float32) / (self.x_dim - 1)
        yy_channel = tf.cast(yy_channel, tf.float32) / (self.y_dim - 1)
        xx_channel = xx_channel*2 - 1
        yy_channel = yy_channel*2 - 1
        ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = tf.math.sqrt(tf.square(xx_channel) + tf.square(yy_channel))
            ret = tf.concat([ret, rr], axis=-1)

        return ret


class CoordConv2D(tf.keras.layers.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self, x_dim, y_dim, with_r, *args, **kwargs):
        super(CoordConv2D, self).__init__()
        self.addcoords = AddCoords(x_dim=x_dim, y_dim=y_dim, with_r=with_r)
        self.conv = tf.keras.layers.Conv2D(*args, **kwargs)

    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret
'''

class AddCoords(tf.keras.layers.Layer):
    """
    Add Coord to tensor
    Alternate implementation, use tf.tile instead of tf.matmul, and x_dim, y_dim is got directly from input_tensor
    """
    def __init__(self, with_r=False):
        super(AddCoords, self).__init__()
        self.with_r = with_r

    def call(self, input_tensor):
        batch_size_tensor, x_dim, y_dim = tf.shape(input_tensor)[:3]
        xx_channel = tf.range(y_dim, dtype=tf.float32)
        xx_channel = tf.expand_dims(xx_channel, 0)
        xx_channel = tf.expand_dims(xx_channel, 0) # shape [1,1,y_dim]
        xx_channel = tf.tile(xx_channel, [batch_size_tensor, x_dim, 1])
        xx_channel = tf.expand_dims(xx_channel, -1)

        yy_channel = tf.range(x_dim, dtype=tf.float32)
        yy_channel = tf.expand_dims(yy_channel, 0)
        yy_channel = tf.expand_dims(yy_channel, -1) # shape [1,x_dim, 1]
        yy_channel = tf.tile(yy_channel, [batch_size_tensor, 1, y_dim])
        yy_channel = tf.expand_dims(yy_channel, -1)

        xx_channel = 2 * xx_channel / (tf.cast(x_dim, dtype=tf.float32) - 1) - 1
        yy_channel = 2 * yy_channel / (tf.cast(y_dim, dtype=tf.float32) - 1) - 1
        ret = tf.concat([input_tensor, xx_channel, yy_channel], axis=-1)

        if self.with_r:
            rr = tf.math.sqrt(tf.square(xx_channel) + tf.square(yy_channel))
            ret = tf.concat([ret, rr], axis=-1)

        return ret

class CoordConv2D(tf.keras.layers.Layer):
    """CoordConv layer as in the paper."""
    def __init__(self, with_r=False, *args, **kwargs):
        super(CoordConv2D, self).__init__()
        self.with_r = with_r
        self.addcoords = AddCoords(with_r=with_r)
        self.conv = tf.keras.layers.Conv2D(*args, **kwargs)

    def call(self, input_tensor):
        ret = self.addcoords(input_tensor)
        ret = self.conv(ret)
        return ret

    def get_config(self):
        config = super(CoordConv2D, self).get_config()
        config['with_r'] = self.with_r
