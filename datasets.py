import random
import numpy as np
import os
import torch
import torch.utils.data as data
from PIL import Image, ImageFilter
import torchvision.transforms as transforms
from torchvision.transforms.functional import pad


def check_path(data_path):
    if data_path.endswith('.txt'):
        with open(data_path, 'r') as txt:
            path_list = [line.rstrip() for line in txt]
    else:
        path_list = [data_path]
    
    return path_list


class CustomDataset(data.Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.trans = get_transform(opt.input_size, training=opt.isTrain)

        #Text Recognizer setting 
        self.CHARS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
        self.CHAR2LABEL = {char: i for i, char in enumerate(self.CHARS)}
        self.num_class = len(self.CHARS)

        self.data_list = {}  # A dict, {'label1(char)': [path1, path2, ...], 'label2': [...], ...}
        for c in self.CHARS:
            self.data_list[c] = []
        self.img_list = []  # A list of dicts, [{'path': img_path, 'label': label(char str)}, ...]

        path_list = check_path(opt.data_path)
        for p in path_list:
            label_path = os.path.join(p, 'label')
            label_files = os.listdir(label_path)
            for f in label_files:
                if f.endswith('.txt'):
                    f_path = os.path.join(label_path, f)
                    with open(f_path, 'r') as txt:
                        label = txt.readline()
                        assert len(label)==1, f"{f} Length of label is large than 1. (Must be equal to 1)"
                    img_name = f.split('.')[0] + '.jpg'
                    img_path = os.path.join(p, 'img', img_name)
                    self.data_list[label].append(img_path)
                    self.img_list.append({'path': img_path, 'label': label})

    
    def __getitem__(self, index):
        anchor = self.img_list[index]
        anc_img = apply_trans(anchor['path'], self.trans)
        label_num = self.CHAR2LABEL[anchor['label']]
        anc_label = torch.zeros(self.num_class)
        anc_label[label_num] = 1.0

        return anc_img, anc_label

    def __len__(self):
        return len(self.img_list)


#=======================================
# Transform funtion (input: PIL Image)
#=======================================

def get_pad(img_size, input_size):
    ow, oh = img_size
    s = input_size / max(img_size)
    nw = int(s*ow)
    nh = int(s*oh)
    padding = _get_padding((nw, nh))
    return torch.tensor(padding)


def _scale(image, target_size, method=Image.BICUBIC):
    ow, oh = image.size
    m = max(ow, oh)
    s = target_size / m
    nw = target_size if ow==m else int(s*ow)
    nh = target_size if oh==m else int(s*oh)
    return image.resize((nw, nh), method)


def _mask(image, ratio):
    w, h = image.size
    mask_size = int(min(h*ratio, w*ratio))
    offset_x = random.randint(0, w-mask_size) if (w-mask_size)>0 else 0
    offset_y = random.randint(0, h-mask_size) if (h-mask_size)>0 else 0
    mask_w = mask_size if (offset_x + mask_size)<w else (w-offset_x)
    mask_h = mask_size if (offset_y + mask_size)<h else (h-offset_y)

    mask_img = Image.new('RGB', (mask_w, mask_h), color = (0,0,0))
    image.paste(mask_img, (offset_x, offset_y))
    return image


def _get_padding(image_size):    
    w, h = image_size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class NewPad(object):
    '''Padding non square image into square.
    '''
    def __init__(self, fill=0, padding_mode='constant'):
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode
        
    def __call__(self, img):
        """
        Args:
            img: PIL Image to be padded.
        Returns:
            PIL Image: Padded image.
        """
        return pad(img, _get_padding(img.size), self.fill, self.padding_mode)
    
    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def get_transform(input_size, method=Image.BICUBIC, training=False):
    '''Transformation list for input.
    '''
    transform_list = []
    
    transform_list.append(transforms.Lambda(lambda img: _scale(img, input_size)))
    transform_list.append(NewPad())

    if training:
        transform_list.append(transforms.RandomPerspective(distortion_scale=0.5, p=0.6))
        transform_list.append(transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.6))
        transform_list.append(transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.3))

    transform_list.append(transforms.Grayscale(3))
    transform_list.append(transforms.ToTensor())
    transform_list.append(transforms.Normalize((0.5,), (0.5,)))
    
    return transforms.Compose(transform_list)


def apply_trans(img_path, trans):
    img = Image.open(img_path)
    img = trans(img)
    return img
