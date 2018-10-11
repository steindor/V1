

class RatioCrop(object):
    """
    Crops the given PIL Image at the center in according to given ratio

    Args:
        ratio (float): Desired output size of the crop. Crops an image
        in the center
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
