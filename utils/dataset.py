import os
import torch
import torch.utils.data # needs to be imported separately
import utils.lib as utils
import numpy
import skimage.transform
from PIL import Image
import torch


# base directory for data
BASE_DATA = './datasets'

# Common extension types used.
TXT_EXT = '.txt'
HDF5_EXT = '.h5'
STATE_EXT = '.pth.tar'
LOG_EXT = '.log'
PNG_EXT = '.png'
PICKLE_EXT = '.pkl'
TEX_EXT = '.tex'
MAT_EXT = '.mat'
GZIP_EXT = '.gz'

# Naming conventions.
def data_file(name, ext=HDF5_EXT):
    """
    Generate path to data file.

    :param name: name of file
    :type name: str
    :param ext: extension (including period)
    :type ext: str
    :return: filepath
    :rtype: str
    """

    return os.path.join(BASE_DATA, name) + ext


def raw_svhn_train_file():
    """
    Raw SVHN training directory.

    :return: fielpath
    :rtype: str
    """

    return data_file('svhn/train_32x32', '.mat')


def raw_svhn_test_file():
    """
    Raw SVHN training directory.

    :return: fielpath
    :rtype: str
    """

    return data_file('svhn/test_32x32', '.mat')


def svhn_train_images_file():
    """
    SVHN train images.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/train_images', HDF5_EXT)


def svhn_test_images_file():
    """
    SVHN test images.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/test_images', HDF5_EXT)


def svhn_train_labels_file():
    """
    SVHN train labels.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/train_labels', HDF5_EXT)


def svhn_test_labels_file():
    """
    SVHN test labels.

    :return: filepath
    :rtype: str
    """

    return data_file('svhn/test_labels', HDF5_EXT)


class CleanDataset(torch.utils.data.Dataset):
    """
    General, clean dataset used for training, testing and attacking.
    """

    def __init__(self, images, labels, indices=None, transform=None):
        """
        Constructor.

        :param images: images/inputs
        :type images: str or numpy.ndarray
        :param labels: labels
        :type labels: str or numpy.ndarray
        :param indices: indices
        :type indices: numpy.ndarray
        :param resize: resize in [channels, height, width
        :type resize: resize
        """

        self.images_file = None
        """ (str) File images were loaded from. """

        self.labels_file = None
        """ (str) File labels were loaded from. """

        if isinstance(images, str):
            self.images_file = images
            images = utils.read_hdf5(self.images_file)
        if not images.dtype == numpy.float32:
            images = images.astype(numpy.float32)

        if isinstance(labels, str):
            self.labels_file = labels
            labels = utils.read_hdf5(self.labels_file)
        labels = numpy.squeeze(labels)
        if not labels.dtype == int:
            labels = labels.astype(int)

        assert isinstance(images, numpy.ndarray)
        assert isinstance(labels, numpy.ndarray)
        assert images.shape[0] == labels.shape[0]

        if indices is None:
            indices = range(images.shape[0])
        assert numpy.min(indices) >= 0
        assert numpy.max(indices) < images.shape[0]

        self.images = images[indices]
        # self.images = []
        # for i in indices:
        #     self.images.append(Image.fromarray((images[i] * 255).astype(numpy.uint8)))
        """ (numpy.ndarray) Inputs. """

        self.labels = labels[indices]
        """ (numpy.ndarray) Labels. """

        self.transform = transform
        """ (numpy.ndarray) Possible attack targets. """

    def __getitem__(self, index):
        assert index < len(self)
        
        if self.transform is None:
            return self.images[index], self.labels[index]
        else:
            return self.transform(self.images[index]), self.labels[index]

    def __len__(self):
        assert len(self.images) == self.labels.shape[0]
        return len(self.images)

    def __add__(self, other):
        return torch.utils.data.ConcatDataset([self, other])


class SVHNTrainSet(CleanDataset):
    def __init__(self, transform=None):
        super(SVHNTrainSet, self).__init__(svhn_train_images_file(), svhn_train_labels_file(), None, transform)


class SVHNTestSet(CleanDataset):
    def __init__(self, transform=None):
        super(SVHNTestSet, self).__init__(svhn_test_images_file(), svhn_test_labels_file(), range(10000), transform)


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            x = self.transform(x)

        return x, y

    def __len__(self):
        return len(self.data)