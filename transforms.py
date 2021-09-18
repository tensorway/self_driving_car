#%%
from torchvision import transforms
import torch.nn.functional as F
import torch
import torch as th

resize_size = (400, 224)
img_size = (224, 224)

class ToIntLabel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return th.round(x/0.0039).type(torch.LongTensor)



class SameRandomCropForNTimes(torch.nn.Module):
    '''
    taken from pythorch RandomCrop transform and 
    added cache to make the same crop for n times
    '''
    def get_params(img, output_size):
        self.cnt = (self.cnt + 1) % self.n
        if self.cnt:
            return self.cache

        w, h = F._get_image_size(img)
        th, tw = output_size

        if h + 1 < th or w + 1 < tw:
            raise ValueError(
                "Required crop size {} is larger then input image size {}".format((th, tw), (h, w))
            )

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1, )).item()
        j = torch.randint(0, w - tw + 1, size=(1, )).item()
        self.cache = i, j, th, tw
        return i, j, th, tw


    def __init__(self, size, padding=None, pad_if_needed=False, fill=0, padding_mode="constant", n=2):
        super().__init__()
        self.cnt = 0
        self.size = size
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode
        self.n = n

    def forward(self, img):
        if self.padding is not None:
            img = F.pad(img, self.padding, self.fill, self.padding_mode)

        width, height = F._get_image_size(img)
        # pad the width if needed
        if self.pad_if_needed and width < self.size[1]:
            padding = [self.size[1] - width, 0]
            img = F.pad(img, padding, self.fill, self.padding_mode)
        # pad the height if needed
        if self.pad_if_needed and height < self.size[0]:
            padding = [0, self.size[0] - height]
            img = F.pad(img, padding, self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)

        return F.crop(img, i, j, h, w)


    def __repr__(self):
        return self.__class__.__name__ + "(size={0}, padding={1})".format(self.size, self.padding)


img_transfrom_base = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

label_transfrom_base = transforms.Compose([
        transforms.ToTensor(),
        ToIntLabel(),
    ])

same_random_crop = SameRandomCropForNTimes(resize_size, n=2)

img_transfrom_train = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(img_size),
    # transforms.RandomHorizontalFlip(), ovo
    # transforms.ColorJitter(),
    # transforms.GaussianBlur(),
    # img_transfrom_base,
    # same_random_crop,
    ])

label_transfrom_train = transforms.Compose([
    transforms.Resize(resize_size),
    transforms.CenterCrop(img_size),
    # transforms.RandomHorizontalFlip(), ovo
    label_transfrom_base,
    # same_random_crop,
    ])

