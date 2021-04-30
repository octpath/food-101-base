# coding: utf-8
import tensorflow as tf
import tensorflow_addons as tfa
import math


def augment():
    @tf.function
    def augment_(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_contrast(img, 0.6, 1.2)
        img = tf.image.random_saturation(img, 0.6, 1.2)
        img = tf.image.random_brightness(img, 0.2)
        return img
    return augment_


def rand_augment(num_augments, magnitude):
    ra = RandAugment()
    @tf.function
    def rand_augment_(img):
        ra.apply(img, num_augments=num_augments, magnitude=magnitude)
        return img
    return rand_augment_


'''
Rand Augment
'''
class RandAugment:
    def __init__(self, max_level=10, cutout_const=80, translate_const=100, replace_value=0):
        self.max_level = max_level
        self.cutout_const = cutout_const
        self.translate_const = translate_const
        self.replace = replace_value
        self.operations = [
            self.identity,
            self.autocontrast,
            self.equalize,
            self.invert,
            self.rotate,
            self.posterize,
            self.solarize,
            self.solarize_add,
            self.color,
            self.contrast,
            self.brightness,
            self.sharpness,
            self.shear_x,
            self.shear_y,
            self.translate_x,
            self.translate_y,
            self.cutout,
        ]
        
    @tf.function
    def apply(self, uint8_image, num_augments=2, magnitude=5, constant=True):
        for _ in range(num_augments):
            op_to_select = tf.random.uniform((), 0, len(self.operations), dtype=tf.int32)
            if constant:
                level = tf.cast(magnitude, tf.float32)
            else:
                level = tf.random.uniform((), 0., tf.cast(magnitude, tf.float32), dtype=tf.float32)
            for i, operation in enumerate(self.operations):
                uint8_image = tf.cond(
                    tf.equal(i, op_to_select),
                    lambda: operation(uint8_image, level),
                    lambda: uint8_image
                )
        return uint8_image
    
    @tf.function
    def _rotate_level_to_arg(self, rel_level):
        level = rel_level * 30.
        level = level if tf.random.normal(()) > 0. else -level
        return level
    
    @tf.function
    def _enhance_level_to_arg(self, rel_level):
        return rel_level * 1.8 + 0.1
    
    @tf.function
    def _shear_level_to_arg(self, rel_level):
        level = rel_level * 0.3
        level = level if tf.random.normal(()) > 0. else -level
        return level
    
    @tf.function
    def _translate_level_to_arg(self, rel_level, translate_const):
        level = rel_level * tf.cast(translate_const, tf.float32)
        level = level if tf.random.normal(()) > 0. else -level
        return level
    
    # -------------------------------------------------------------------------------------
    
    @tf.function
    def identity(self, image, level):
        return image
    
    @tf.function
    def autocontrast(self, image, level):
        @tf.function
        def scale_channel(image):
            """Scale the 2D image using the autocontrast rule."""
            # A possibly cheaper version can be done using cumsum/unique_with_counts
            # over the histogram values, rather than iterating over the entire image.
            # to compute mins and maxes.
            lo = tf.cast(tf.reduce_min(image), tf.float32)
            hi = tf.cast(tf.reduce_max(image), tf.float32)
            # Scale the image, making the lowest value 0 and the highest value 255.
            def scale_values(im):
                scale = 255.0 / (hi - lo)
                offset = -lo * scale
                im = tf.cast(im, tf.float32) * scale + offset
                im = tf.clip_by_value(im, 0.0, 255.0)
                return tf.cast(im, tf.uint8)
            result = tf.cond(hi > lo, lambda: scale_values(image), lambda: image)
            return result
        s1 = scale_channel(image[:, :, 0])
        s2 = scale_channel(image[:, :, 1])
        s3 = scale_channel(image[:, :, 2])
        image = tf.stack([s1, s2, s3], axis=-1)
        return image
    
    @tf.function
    def equalize(self, image, level):
        return tfa.image.equalize(image)
    
    @tf.function
    def invert(self, image, level):
        return tf.bitwise.invert(image)
    
    @tf.function
    def rotate(self, image, level):
        degree = self._rotate_level_to_arg(level/self.max_level)
        radian = degree * math.pi / 180.0
        return tfa.image.rotate(image, radian)
    
    @tf.function
    def posterize(self, image, level):
        bits = tf.cast((level/self.max_level) * 4, tf.uint8)
        shift = 8 - bits
        return tf.bitwise.left_shift(tf.bitwise.right_shift(image, shift), shift)
    
    @tf.function
    def solarize(self, image, level):
        threshold = tf.cast((level/self.max_level) * 256, tf.uint8)
        return tf.where(image < threshold, image, 255 - image)
    
    @tf.function
    def solarize_add(self, image, level):
        addition = tf.cast((level/self.max_level) * 110, tf.int64)
        added_image = tf.cast(image, tf.int64) + addition
        added_image = tf.cast(tf.clip_by_value(added_image, 0, 255), tf.uint8)
        return tf.where(image < 128, added_image, image)
    
    @tf.function
    def color(self, image, level):
        factor = self._enhance_level_to_arg(level/self.max_level)
        degenerate = tf.image.grayscale_to_rgb(tf.image.rgb_to_grayscale(image))
        return tf.cast(tfa.image.blend(tf.cast(degenerate, tf.float32), tf.cast(image, tf.float32), factor), tf.uint8)
    
    @tf.function
    def contrast(self, image, level):
        factor = self._enhance_level_to_arg(level/self.max_level)
        """Equivalent of PIL Contrast."""
        degenerate = tf.image.rgb_to_grayscale(image)
        # Cast before calling tf.histogram.
        degenerate = tf.cast(degenerate, tf.int32)
        # Compute the grayscale histogram, then compute the mean pixel value,
        # and create a constant image size of that value.  Use that as the
        # blending degenerate target of the original image.
        hist = tf.histogram_fixed_width(degenerate, [0, 255], nbins=256)
        mean = tf.reduce_sum(tf.cast(hist, tf.float32)) / 256.0
        degenerate = tf.ones_like(degenerate, dtype=tf.float32) * mean
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.image.grayscale_to_rgb(tf.cast(degenerate, tf.uint8))
        return tf.cast(tfa.image.blend(tf.cast(degenerate, tf.float32), tf.cast(image, tf.float32), factor), tf.uint8)
    
    @tf.function
    def brightness(self, image, level):
        factor = self._enhance_level_to_arg(level/self.max_level)
        degenerate = tf.zeros_like(image)
        return tf.cast(tfa.image.blend(tf.cast(degenerate, tf.float32), tf.cast(image, tf.float32), factor), tf.uint8)
    
    @tf.function
    def sharpness(self, image, level):
        factor = self._enhance_level_to_arg(level/self.max_level)
        """Implements Sharpness function from PIL using TF ops."""
        orig_image = image
        image = tf.cast(image, tf.float32)
        # Make image 4D for conv operation.
        image = tf.expand_dims(image, 0)
        # SMOOTH PIL Kernel.
        kernel = tf.constant(
          [[1, 1, 1], [1, 5, 1], [1, 1, 1]], dtype=tf.float32,
          shape=[3, 3, 1, 1]
        ) / 13.
        # Tile across channel dimension.
        kernel = tf.tile(kernel, [1, 1, 3, 1])
        strides = [1, 1, 1, 1]
        with tf.device('/cpu:0'):
            # Some augmentation that uses depth-wise conv will cause crashing when
            # training on GPU. See (b/156242594) for details.
            degenerate = tf.nn.depthwise_conv2d(
                image, kernel, strides, padding='VALID', dilations=[1, 1]
            )
        degenerate = tf.clip_by_value(degenerate, 0.0, 255.0)
        degenerate = tf.squeeze(tf.cast(degenerate, tf.uint8), [0])
        # For the borders of the resulting image, fill in the values of the
        # original image.
        mask = tf.ones_like(degenerate)
        padded_mask = tf.pad(mask, [[1, 1], [1, 1], [0, 0]])
        padded_degenerate = tf.pad(degenerate, [[1, 1], [1, 1], [0, 0]])
        result = tf.where(tf.equal(padded_mask, 1), padded_degenerate, orig_image)
        # Blend the final result.
        return tf.cast(tfa.image.blend(tf.cast(result, tf.float32), tf.cast(orig_image, tf.float32), factor), tf.uint8)
    
    @tf.function
    def shear_x(self, image, level):
        level = self._shear_level_to_arg(level/self.max_level)
        return tfa.image.shear_x(image, level, self.replace)
    
    @tf.function
    def shear_y(self, image, level):
        level = self._shear_level_to_arg(level/self.max_level)
        return tfa.image.shear_y(image, level, self.replace)
    
    @tf.function
    def cutout(self, image, level):
        mask_size = int((level/self.max_level) * self.cutout_const)
        return tfa.image.random_cutout(tf.expand_dims(image, 0), mask_size, self.replace)[0]
    
    @tf.function
    def translate_x(self, image, level):
        pixels = self._translate_level_to_arg(level/self.max_level, self.translate_const)
        return tfa.image.translate_xy(image, [-pixels, 0], self.replace)
    
    @tf.function
    def translate_y(self, image, level):
        pixels = self._translate_level_to_arg(level/self.max_level, self.translate_const)
        return tfa.image.translate_xy(image, [0, -pixels], self.replace)
