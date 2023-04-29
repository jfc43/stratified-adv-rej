from imgaug import augmenters as iaa
from imgaug import dtypes as iadt
import numpy

class Clip(iaa.Augmenter):
    """
    Clip augmenter.
    """

    def __init__(self, min=0, max=1, name=None, deterministic=False, random_state=None):
        super(Clip, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

        self.min = min
        """ (float) Minimum."""

        self.max = max
        """ (float) Maximum. """

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images, allowed=["float32"], disallowed=[
            "bool", "uint8", "uint16",
            "int8", "int16", "float16",
            "uint32", "uint64", "uint128",
            "uint256", "int32", "int64",
            "int128", "int256", "float64",
            "float96", "float128", "float256"
        ], augmenter=self)

        converted_images = numpy.clip(images, self.min, self.max)
        return converted_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []


class Image(iaa.Augmenter):
    """
    Image augmenter.
    """

    def __init__(self, name=None, deterministic=False, random_state=None):
        super(Image, self).__init__(name=name, deterministic=deterministic, random_state=random_state)

    def _augment_images(self, images, random_state, parents, hooks):
        iadt.gate_dtypes(images, allowed=["float32"], disallowed=[
            "bool", "uint8", "uint16",
            "int8", "int16", "float16",
            "uint32", "uint64", "uint128",
            "uint256", "int32", "int64",
            "int128", "int256", "float64",
            "float96", "float128", "float256"
        ], augmenter=self)

        converted_images = (images * 255).astype(numpy.uint8)
        return converted_images

    def _augment_heatmaps(self, heatmaps, random_state, parents, hooks):
        return heatmaps

    def _augment_keypoints(self, keypoints_on_images, random_state, parents, hooks):
        return keypoints_on_images

    def get_parameters(self):
        return []