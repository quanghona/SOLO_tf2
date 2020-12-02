import tensorflow as tf


class FeaturePyramidNetwork(tf.keras.layers.Layer):
    """
    Creating the backbone component of feature Pyramid Network
    paper: https://arxiv.org/pdf/1612.03144.pdf
    This is a slightly more general version than original paper

    Can be any number of input feature maps.
    Can have down sampled output feature maps
    """

    def __init__(self, num_channel, num_feature_map=3, num_downsample=0, **kwargs):
        """Creating FPN instance

        Args:
            num_channel (int): number of filter channels
            num_feature_map (int, optional): number of input feature map
            num_downsample (int, optional): number of down sampled feature maps in pyramid. Defaults to 0.
        """
        super(FeaturePyramidNetwork, self).__init__(**kwargs)

        self.num_channel = num_channel
        self.num_feature_map = num_feature_map
        self.num_downsample = num_downsample

        self.Add = tf.keras.layers.Add()
        self.upSample = tf.keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')
        self.downSamples = [None] * num_downsample
        for i in range(num_downsample):
            self.downSamples[i] = tf.keras.layers.Conv2D(num_channel, (3, 3), 2, padding="same",
                                                        kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.lateralConvs = [None] * num_feature_map
        for i in range(num_feature_map):
            self.lateralConvs[i] = tf.keras.layers.Conv2D(num_channel, (1, 1), 1, padding="same",
                                                          kernel_initializer=tf.keras.initializers.glorot_uniform())

        self.predictConvs = [None] * num_feature_map
        for i in range(num_feature_map):
            self.predictConvs[i] = tf.keras.layers.Conv2D(num_channel, (3, 3), 1, padding="same",
                                                       kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                       activation='relu')


    def call(self, Cx):
        """Build the FPN network

        Args:
            Cx (list): list of input feature maps, ascending order (e.g. C2, C3, C4...)
            training (bool, optional): [description]. Defaults to False.

        Returns:
            list: list of feature map, which follow the same order of Cx
        """
        Px = [None] * self.num_feature_map

        # Attach lateral convolutional layer on last feature map
        c_last = Cx.pop()
        Px[-1] = self.lateralConvs[-1](c_last)

        # Merges lateral connections with input feature maps
        for i,c in reversed(list(enumerate(Cx))):
            Px[i] = self._crop_and_add(self.upSample(Px[i+1]), self.lateralConvs[i](c))

        # Generate final output feature map
        for i in range(self.num_feature_map):
            Px[i] = self.predictConvs[i](Px[i])

        # Generate down sampled feature maps (if any)
        P_downsample = Px[-1]
        for i in range(self.num_downsample):
            P_downsample = self.downSamples[i](P_downsample)
            Px.append(P_downsample)

        return Px

    def _crop_and_add(self, x1, x2):
        """for p4, c4; p3, c3 to concatenate with matched shape
        https://tf-unet.readthedocs.io/en/latest/_modules/tf_unet/layers.html

        Args:
            x1 (Tensor): input
            x2 (Tensor): input

        Returns:
            [Tensor]: Result Added tensor
        """
        x1_shape = x1.shape
        x2_shape = x2.shape
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return self.Add([x1_crop, x2])

    def get_config(self):
        config = super(FeaturePyramidNetwork, self).get_config()
        config['num_channel'] = self.num_channel
        config['num_feature_map'] = self.num_feature_map
        config['num_downsample'] = self.num_downsample
        return config
