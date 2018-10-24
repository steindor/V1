import torch
import torchvision.transforms.functional as F
from IPython.core.debugger import set_trace
import types
import numpy as np
import random
import math
import numbers
from PIL import Image, ImageOps

class RatioCrop(object):
    """
    Crops the given PIL Image in the center randomly between ratio and 1.0 

    Args:
        ratio (float): Desired lower limig of the output size of the crop. 
    """

    def __init__(self, ratio, random=False):
        self.ratio = ratio
        self.random = random

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.
            random (bool): If passed as true, a random crop between ratio and 1.0 is executed

        Returns:
            PIL Image: Cropped image.
        """
        width, height = img.size

        if not self.random:
            self.size = (width*self.ratio, height*self.ratio)
        else:
            random_ratio = random.uniform(self.ratio, 1.0)
            self.size = (width*random_ratio, height*random_ratio)

        return F.center_crop(img, self.size)


class ChangeBrightness(object):

    def __init__(self, brightness):
        self.brightness = brightness

    def __call__(self, img):
        return F.adjust_brightness(img, self.brightness)


""" 

    Transforms that apply to segmentation tasks, img and mask

"""


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        
        for t in self.transforms:
            img, mask = t(img, mask)
        
        return img, mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

class singleCompose(object):
    
    """Composes several transforms together for only an image, not a mask

    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

# class RandomCrop(object):
#     def __init__(self, size, padding=0):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size  # h, w
#         self.padding = padding

#     def __call__(self, img, mask):

#         if self.padding > 0:
#             img = ImageOps.expand(img, border=self.padding, fill=0)
#             mask = ImageOps.expand(mask, border=self.padding, fill=0)

#         assert img.size == mask.size
#         w, h = img.size
#         th, tw = self.size  # target size
#         if w == tw and h == th:
#             return {'image': img,
#                     'label': mask}
#         if w < tw or h < th:
#             img = img.resize((tw, th), Image.BILINEAR)
#             mask = mask.resize((tw, th), Image.NEAREST)
#             return {'image': img,
#                     'label': mask}

#         x1 = random.randint(0, w - tw)
#         y1 = random.randint(0, h - th)
#         img = img.crop((x1, y1, x1 + tw, y1 + th))
#         mask = mask.crop((x1, y1, x1 + tw, y1 + th))

#         return img, mask


# class CenterCrop(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         assert img.size == mask.size
#         w, h = img.size
#         th, tw = self.size
#         x1 = int(round((w - tw) / 2.))
#         y1 = int(round((h - th) / 2.))
#         img = img.crop((x1, y1, x1 + tw, y1 + th))
#         mask = mask.crop((x1, y1, x1 + tw, y1 + th))

#         return {'image': img,
#                 'label': mask}


class RandomHorizontalFlip(object):

    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, mask):

        if random.random() < self.p:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)

        return img, mask

class RandomVerticalFlip(object):

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, mask):

        random_n = random.random()

        if random_n < self.p:
            img = F.vflip(img)
        
        if random_n < self.p:
            mask = F.vflip(mask)

        return img, mask



# class Normalize(object):
#     """Normalize a tensor image with mean and standard deviation.
#     Args:
#         mean (tuple): means for each channel.
#         std (tuple): standard deviations for each channel.
#     """

#     def __init__(self, mean=(0., 0., 0.), std=(1., 1., 1.)):
#         self.mean = mean
#         self.std = std

#     def __call__(self, sample):
#         img = np.array(sample['image']).astype(np.float32)
#         mask = np.array(sample['label']).astype(np.float32)
#         img /= 255.0
#         img -= self.mean
#         img /= self.std

#         return {'image': img,
#                 'label': mask}


# class FixedResize(object):
#     def __init__(self, size):
#         self.size = tuple(reversed(size))  # size: (h, w)

#     def __call__(self, img, mask):
        
#         assert img.size == mask.size

#         img = img.resize(self.size, Image.BILINEAR)
#         mask = mask.resize(self.size, Image.NEAREST)

#         return img, mask


# class Scale(object):
#     def __init__(self, size):
#         if isinstance(size, numbers.Number):
#             self.size = (int(size), int(size))
#         else:
#             self.size = size

#     def __call__(self, img, mask):

#         assert img.size == mask.size
#         w, h = img.size

#         if (w >= h and w == self.size[1]) or (h >= w and h == self.size[0]):
#             return {'image': img,
#                     'label': mask}
#         oh, ow = self.size
#         img = img.resize((ow, oh), Image.BILINEAR)
#         mask = mask.resize((ow, oh), Image.NEAREST)

#         return img, mask


# class RandomSizedCrop(object):
#     def __init__(self, size):
#         self.size = size

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         assert img.size == mask.size
#         for attempt in range(10):
#             area = img.size[0] * img.size[1]
#             target_area = random.uniform(0.45, 1.0) * area
#             aspect_ratio = random.uniform(0.5, 2)

#             w = int(round(math.sqrt(target_area * aspect_ratio)))
#             h = int(round(math.sqrt(target_area / aspect_ratio)))

#             if random.random() < 0.5:
#                 w, h = h, w

#             if w <= img.size[0] and h <= img.size[1]:
#                 x1 = random.randint(0, img.size[0] - w)
#                 y1 = random.randint(0, img.size[1] - h)

#                 img = img.crop((x1, y1, x1 + w, y1 + h))
#                 mask = mask.crop((x1, y1, x1 + w, y1 + h))
#                 assert (img.size == (w, h))

#                 img = img.resize((self.size, self.size), Image.BILINEAR)
#                 mask = mask.resize((self.size, self.size), Image.NEAREST)

#                 return {'image': img,
#                         'label': mask}

        # Fallback
        # scale = Scale(self.size)
        # crop = CenterCrop(self.size)
        # sample = crop(scale(sample))
        # return sample


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):

        rotate_degree = random.random() * 2 * self.degree - self.degree
        img = img.rotate(rotate_degree, Image.BILINEAR)
        mask = mask.rotate(rotate_degree, Image.NEAREST)

        return img, mask


class Lambda(object):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'

class ColorJitter(object):
    """Randomly change the brightness, contrast and saturation of an image.

    Args:
        brightness (float): How much to jitter brightness. brightness_factor
            is chosen uniformly from [max(0, 1 - brightness), 1 + brightness].
        contrast (float): How much to jitter contrast. contrast_factor
            is chosen uniformly from [max(0, 1 - contrast), 1 + contrast].
        saturation (float): How much to jitter saturation. saturation_factor
            is chosen uniformly from [max(0, 1 - saturation), 1 + saturation].
        hue(float): How much to jitter hue. hue_factor is chosen uniformly from
            [-hue, hue]. Should be >=0 and <= 0.5.
    """

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """

        transforms = []
        if brightness > 0:
            brightness_factor = random.uniform(
                max(0, 1 - brightness), 1 + brightness)
            transforms.append(
                Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast > 0:
            contrast_factor = random.uniform(
                max(0, 1 - contrast), 1 + contrast)
            transforms.append(
                Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation > 0:
            saturation_factor = random.uniform(
                max(0, 1 - saturation), 1 + saturation)
            transforms.append(
                Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue > 0:
            hue_factor = random.uniform(-hue, hue)
            transforms.append(
                Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = singleCompose(transforms)

        return transform

    def __call__(self, img, mask):
        """
        Args:
            img (PIL Image): Input image.

        Returns:
            PIL Image: Color jittered image.
        """
        transform = self.get_params(self.brightness, self.contrast,
                                    self.saturation, self.hue)
        # no need to transform mask
        return transform(img), mask

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        return format_string

class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        w = int(random.uniform(0.8, 2.5) * img.size[0])
        h = int(random.uniform(0.8, 2.5) * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize(
            (w, h), Image.NEAREST)
        sample = {'image': img, 'label': mask}

        return self.crop(self.scale(sample))


class RandomScale(object):
    def __init__(self, limit):
        self.limit = limit

    def __call__(self, sample):
        img = sample['image']
        mask = sample['label']
        assert img.size == mask.size

        scale = random.uniform(self.limit[0], self.limit[1])
        w = int(scale * img.size[0])
        h = int(scale * img.size[1])

        img, mask = img.resize((w, h), Image.BILINEAR), mask.resize(
            (w, h), Image.NEAREST)

        return {'image': img, 'label': mask}
