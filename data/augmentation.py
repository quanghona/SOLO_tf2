import tensorflow as tf


"""
Implement the augmentation to the image and labels (contains categories and mask)

Note: The accepted image pass to the function is scaled to the range [0, 1]
Image and labels should have type tf.float32

"""

def randomBrightness(image, cat=None, mask=None, max_delta=0.2):
    """randomly change the brigthness of the image

    Args:
        image (tf.Tensor): tensor image object
        cat (tf.Tensor, optional): Category label tensor. Defaults to None.
        mask (tf.Tensor, optional): Mask label tensor. Defaults to None.
        max_delta (float, optional): adjust factor, range [0, 1]. Defaults to 0.2.

    Returns:
        tf.Tensor: Adjusted image tensor
        tf.Tensor: Category label tensor (same as input)
        tf.Tensor: Mask label tensor (same as input)
    """
    image = tf.image.random_brightness(image, max_delta)
    return image, cat, mask

def randomContrast(image, cat=None, mask=None, lower=0.4, upper=0.6):
    """Randomly change the contrast of the image

    Args:
        image (tf.Tensor): tensor image object
        cat (tf.Tensor, optional): Category label tensor. Defaults to None.
        mask (tf.Tensor, optional): Mask label tensor. Defaults to None.
        lower (float, optional): lower bound of contrast factor. Defaults to 0.4.
        upper (float, optional): upper bound of contrast factor. Defaults to 0.6.

    Returns:
        tf.Tensor: Adjusted image tensor
        tf.Tensor: Category label tensor (same as input)
        tf.Tensor: Mask label tensor (same as input)
    """
    image = tf.image.random_contrast(image, lower, upper)
    return image, cat, mask

def flipHorizontal(image, cat, mask):
    """Flip the image and label horizontally
    Note: this function flip the category and mask also

    Args:
        image (tf.Tensor): tensor image object
        cat (tf.Tensor): Category label tensor
        mask (tf.Tensor): Maks label tensor

    Returns:
        tf.Tensor: Adjusted image tensor
        tf.Tensor: Category label tensor
        tf.Tensor: Mask label tensor
    """
    image = tf.image.flip_left_right(image)
    cat = tf.image.flip_left_right(cat)
    mask = tf.image.flip_left_right(mask)
    return image, cat, mask

def randomHUE(image, cat=None, mask=None, max_delta=0.2):
    """Randomly change the HUE of the image

    Args:
        image (tf.Tensor): tensor image object
        cat (tf.Tensor, optional): Category label tensor. Defaults to None.
        mask (tf.Tensor, optional): Mask label tensor. Defaults to None.
        max_delta (float, optional): Adjsut factor. Defaults to 0.2.

    Returns:
        tf.Tensor: Adjusted image tensor
        tf.Tensor: Category label tensor (same as input)
        tf.Tensor: Mask label tensor (same as input)
    """
    image = tf.image.random_hue(image, max_delta)
    return image, cat, mask

def randomSaturation(image, cat=None, mask=None, lower=0.4, upper=0.6):
    """Randomly change the saturation of the image

    Args:
        image (tf.Tensor): tensor image object
        cat (tf.Tensor, optional): Category label tensor. Defaults to None.
        mask (tf.Tensor, optional): Mask label tensor. Defaults to None.
        lower (float, optional): lower bound of saturation factor. Defaults to 0.4.
        upper (float, optional): upper bound of saturation factor. Defaults to 0.6.

    Returns:
        tf.Tensor: Adjusted image tensor
        tf.Tensor: Category label tensor (same as input)
        tf.Tensor: Mask label tensor (same as input)
    """
    image = tf.image.random_saturation(image, lower, upper)
    return image, cat, mask
