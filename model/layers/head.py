import tensorflow as tf
from model.layers.coordconv import CoordConv2D


class PredictionHead(tf.keras.layers.Layer):
    """
    Prediction head for SOLO network
    """
    def __init__(self, num_class, grid_sizes, image_size, head_depth=8, num_channel=256, head_style='vanilla', **kwargs):
        """Prediction head module initialization
        Prepare convolutions for category branch and mask branch

        Args:
            num_class (int): number of output class (category)
            grid_sizes (int): list of grid number (denoted S in the paper)
            image_size (int): Original image size, the target size of the mask branch output during inference
            head_depth (int, optional): depth of branc(s). Defaults to 8.
            num_channel (int, optional): number of channels in convolution layers. Defaults to 256.
            head_style (str, optional): Style of head, one of 'vanilla' or 'decoupled'. Defaults to 'vanilla'.
        """
        super(PredictionHead, self).__init__(**kwargs)
        self.num_class = num_class
        self.grid_sizes = grid_sizes
        self.image_size = image_size
        self.head_depth = head_depth
        self.num_channel = num_channel
        self.head_style = head_style
        self.cat_convs = []
        self.mask_convs = []
        self.mask_conv_out = []

        # Categoy branch
        for i in range(head_depth - 1):
            cat_conv = tf.keras.layers.Conv2D(self.num_channel, (3,3), 1, padding="same",
                                             kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                             activation="relu")
            self.cat_convs.append(cat_conv)
        cat_conv_out = tf.keras.layers.Conv2D(self.num_class, (3,3), 1, padding="same",
                                                kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                activation="sigmoid")
        self.cat_convs.append(cat_conv_out)

        # Mask branch
        if head_style == 'vanilla':
            num_mask_branch = 1
        elif head_style == 'decoupled':
            num_mask_branch = 2
        else:
            raise ValueError(f"Head style {head_style} not supported")
        mask_conv_channels = [[self.num_channel, self.num_channel]] * num_mask_branch # number of channels for all sub-branchs in mask branchs

        for branch_channels in mask_conv_channels:
            branch_conv = []
            branch_out = {grid_size: None for grid_size in grid_sizes}
            coordconv = CoordConv2D(filters=branch_channels[0], kernel_size=(3,3), strides=1, padding="same",
                                    kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                    activation="relu")
            branch_conv.append(coordconv)
            for i in range(head_depth - 2):
                mask_conv = tf.keras.layers.Conv2D(branch_channels[1], (3,3), 1, padding="same",
                                                   kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                   activation="relu")
                branch_conv.append(mask_conv)

            for grid_size in grid_sizes:
                mask_out_num_channel = grid_size*grid_size if head_style == "vanilla" else grid_size
                branch_out[grid_size] = tf.keras.layers.Conv2D(mask_out_num_channel, (3,3), 1, padding="same",
                                                               kernel_initializer=tf.keras.initializers.glorot_uniform(),
                                                               activation="sigmoid")
            self.mask_convs.append(branch_conv)
            self.mask_conv_out.append(branch_out)


    def call(self, feature_map, grid_number, training=False):
        # Category branch
        # Alignment process: Use bilinear interpolation
        cat = tf.image.resize(feature_map, [grid_number, grid_number])
        for CatConv2D in self.cat_convs:
            cat = CatConv2D(cat)

        # Mask branch
        mask = []
        for i, branch in enumerate(self.mask_convs):
            mask_branch = feature_map
            for MaskConv2D in branch:
                mask_branch = MaskConv2D(mask_branch)
            mask_branch = self.mask_conv_out[i][grid_number](mask_branch)
            if training is False:   # Upsample to original image size
                mask_branch = tf.image.resize(mask_branch, [self.image_size, self.image_size])
            mask.append(mask_branch)

        return cat, mask


    def get_config(self):
        config = super(PredictionHead, self).get_config()
        config['num_class'] = self.num_class
        config['grid_sizes'] = self.grid_sizes
        config['head_depth'] = self.head_depth
        config['num_channel'] = self.num_channel
        config['head_style'] = self.head_style
        return config
