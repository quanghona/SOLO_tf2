import tensorflow as tf

from model.layers.fpn import FeaturePyramidNetwork
from model.layers.head import PredictionHead

assert tf.__version__.startswith('2')


class SOLO(tf.keras.Model):
    def __init__(self, num_class, input_size, grid_sizes=[24], backbone="resnet50", head_style="vanilla", head_depth=8, fpn_channel=256, **kwargs):
        super(SOLO, self).__init__(**kwargs)
        self.num_class = num_class
        self.input_size = input_size
        self.grid_sizes = grid_sizes
        self.backbone = backbone
        self.head_style = head_style
        self.head_depth = head_depth
        self.fpn_channel = fpn_channel

        if backbone == 'resnet50':
            self.backbone_out_layers = ['conv3_block4_out', 'conv4_block6_out', 'conv5_block3_out']
            base_model = tf.keras.applications.ResNet50(input_shape=(input_size, input_size, 3),
                                                        include_top=False,
                                                        layers=tf.keras.layers,
                                                        weights=None)
        elif backbone == 'mobilenetv2':
            self.backbone_out_layers = ['block_6_expand_relu', 'block_13_expand_relu', 'block_16_project']
            base_model = tf.keras.applications.MobileNetV2(input_shape=(input_size, input_size, 3),
                                                           include_top=False,
                                                           weights=None)
        elif backbone == 'mobilenet':
            self.backbone_out_layers = ['conv_pw_5_relu', 'conv_pw_11_relu', 'conv_pw_13_relu']
            base_model = tf.keras.applications.MobileNet(input_shape=(input_size, input_size, 3),
                                                         include_top=False,
                                                         weights=None)
        elif backbone == 'xception':
            self.backbone_out_layers = ['block4_sepconv2', 'block13_sepconv2', 'block14_sepconv2_act']
            base_model = tf.keras.applications.Xception(input_shape=(input_size, input_size, 3),
                                                        include_top=False,
                                                        weights=None)
        else:
            raise NotImplementedError('Backbone %s not supported' % (backbone))

        self.backbone = tf.keras.Model(inputs=base_model.input,
                                       outputs=[base_model.get_layer(x).output for x in self.backbone_out_layers], name=backbone)
        self.fpn = FeaturePyramidNetwork(fpn_channel, num_feature_map=len(self.backbone_out_layers))
        self.head = PredictionHead(num_class, grid_sizes, input_size, head_style=head_style, head_depth=head_depth, num_channel=fpn_channel)


    def call(self, input, training=False):
        Cx = self.backbone(input, training=training)

        Px = self.fpn(Cx)

        cat_out = []
        mask_out = []
        for p,grid_number in zip(Px, self.grid_sizes):
            cat, mask = self.head(p, grid_number, training=training)
            cat_out.append(cat)
            mask_out.append(mask)

        # cat_out = tf.concat(cat_out, axis=1)
        # mask_out = tf.concat(mask_out, axis=1)

        return cat_out, mask_out


    def get_config(self):
        config = super(SOLO, self).get_config()
        config['num_class'] = self.num_class,
        config['input_size'] = self.input_size,
        config['grid_sizes'] = self.grid_sizes,
        config['backbone'] = self.backbone,
        config['backbone_out_layers'] = self.backbone_out_layers,
        config['head_style'] = self.head_style,
        config['head_depth'] = self.head_depth,
        config['fpn_channel'] = self.fpn_channel
        return config
