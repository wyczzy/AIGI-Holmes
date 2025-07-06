import cv2
import numpy as np
from albumentations import Downscale, Compose, ImageCompression, GaussNoise, MotionBlur, GaussianBlur
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF
from random import random, choice
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from scipy.ndimage.filters import gaussian_filter
from torchvision.transforms import InterpolationMode

ImageFile.LOAD_TRUNCATED_IMAGES = True

def dataset_folder(opt, root):
    if opt.mode == 'binary':
        return binary_dataset(opt, root)
    if opt.mode == 'filename':
        return FileNameDataset(opt, root)
    raise ValueError('opt.mode needs to be binary or filename.')
    

import base64
from io import BytesIO
from PIL import Image
import numpy as np
import torchvision.transforms
from scipy.ndimage.filters import gaussian_filter


class ImageProcessor:
    def __init__(self, opt):
        self.opt = opt

    def data_augment(self, img):
        noise_type = getattr(self.opt, "noise_type", None)
        noise_ratio = getattr(self.opt, "noise_ratio", None)

        if noise_type == 'jpg':
            img_processed = self.pil_jpg_new(img, noise_ratio)
        elif noise_type == 'resize':
            width, height = img.size
            img_processed = torchvision.transforms.Resize((int(height / noise_ratio), int(width / noise_ratio)))(img)
        elif noise_type == 'blur':
            img = np.array(img)
            self.gaussian_blur(img, noise_ratio)
            img_processed = Image.fromarray(img)
        elif noise_type is None:
            img_processed = img

        return img_processed

    def pil_jpg_new(self, img, compress_val):
        out = BytesIO()
        img.save(out, format='jpeg', quality=compress_val)
        img = Image.open(out)
        # load from memory before ByteIO closes
        img = np.array(img)
        img = Image.fromarray(img)
        out.close()
        return img

    def gaussian_blur(self, img, sigma):
        # Check if the image has a third dimension
        if img.ndim == 3:
            # Check the number of channels in the image
            channels = img.shape[2]
            # if channels == 0:
            #     # If there are no channels, return the image as is
            #     return img
            if channels == 1:
                # If there is one channel, apply the filter to the single channel
                gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
            elif channels == 2:
                # If there are two channels, apply the filter to both channels
                gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
                gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
            elif channels == 3:
                # If there are three channels, apply the filter to all three channels
                gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
                gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
                gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)
        else:
            # If the image does not have a third dimension, apply the filter to the 2D image
            gaussian_filter(img, output=img, sigma=sigma)

        # return img


def encode_image(image_path, processor=None):
    with open(image_path, "rb") as image_file:
        img = Image.open(image_file)

        # Apply data augmentation if processor is provided
        if processor is not None:
            # print(processor.opt)
            img = processor.data_augment(img)

        # Convert image to Base64
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return img, img_base64

class Options:
    def __init__(self, noise_type=None, noise_ratio=None):
        """
        初始化数据增强的配置参数。

        :param noise_type: 噪声类型，可选值为 'jpg', 'resize', 'blur' 或 None
        :param noise_ratio: 噪声强度或比例，具体含义取决于 noise_type
        """
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
    def __str__(self):
        """返回对象的字符串表示，便于调试"""
        return (f"Options(noise_type={self.noise_type}, noise_ratio={self.noise_ratio}")


# Example usage:
# opt = SomeOptionsClass()  # Replace with your actual options class
# print(opt_class)


def binary_dataset(opt, root):
    if opt.isTrain:
        crop_func = transforms.RandomCrop(opt.cropSize)
    elif opt.no_crop:
        crop_func = transforms.Lambda(lambda img: img)
    else:
        crop_func = transforms.CenterCrop(opt.cropSize)

    if opt.isTrain and not opt.no_flip:
        flip_func = transforms.RandomHorizontalFlip()
    else:
        flip_func = transforms.Lambda(lambda img: img)
    if not opt.isTrain and opt.no_resize:
        rz_func = transforms.Lambda(lambda img: img)
    else:
        # rz_func = transforms.Lambda(lambda img: custom_resize(img, opt))
        rz_func = transforms.Resize((opt.loadSize, opt.loadSize))
        
    NOISE_TYPE = getattr(opt, "noise_type", None)
    NOISE_RATIO = getattr(opt, "noise_ratio", None)
        
    opt_class = Options(noise_type=NOISE_TYPE, noise_ratio=NOISE_RATIO)
    processor = ImageProcessor(opt_class)
    # import pdb; pdb.set_trace()
    
    # 定义Albumentations增强
    aug = Compose([
        ImageCompression(quality_lower=60, quality_upper=100, p=0.5),
        GaussNoise(p=0.2),
        MotionBlur(p=0.2),
        GaussianBlur(blur_limit=3, p=0.5),
        Downscale(scale_min=0.25, scale_max=0.75, interpolation=cv2.INTER_LINEAR, p=0.5)
    ])
    
    # 自定义转换函数：将PIL图像转换为NumPy数组，应用Albumentations增强，再转回PIL
    def albumentations_aug(img):
        # 将PIL图像转换为NumPy数组 (H, W, C) RGB格式
        img_np = np.array(img)
        # 应用Albumentations增强
        augmented = aug(image=img_np)
        # 从字典中提取增强后的图像
        img_aug = augmented['image']
        # 将NumPy数组转回PIL图像
        return Image.fromarray(img_aug)

    DATA_AUG = getattr(opt, "data_aug", False)
    # 创建数据集
    if DATA_AUG:
        data_aug = transforms.Lambda(albumentations_aug)  # 替换点
    else:
        data_aug = transforms.Lambda(lambda img: img)

    # import pdb; pdb.set_trace()
    dset = datasets.ImageFolder(
        root,
        transforms.Compose([
            transforms.Lambda(lambda img: processor.data_augment(img)),
            # 使用自定义函数替换原processor.data_augment
            data_aug,
            rz_func,         # 后续PyTorch变换 (如调整大小)
            crop_func,       # 裁剪
            flip_func,       # 翻转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    )
    return dset


class FileNameDataset(datasets.ImageFolder):
    def name(self):
        return 'FileNameDataset'

    def __init__(self, opt, root):
        self.opt = opt
        super().__init__(root)

    def __getitem__(self, index):
        # Loading sample
        path, target = self.samples[index]
        return path


def data_augment(img, opt):
    img = np.array(img)

    if random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


# rz_dict = {'bilinear': Image.BILINEAR,
           # 'bicubic': Image.BICUBIC,
           # 'lanczos': Image.LANCZOS,
           # 'nearest': Image.NEAREST}
rz_dict = {'bilinear': InterpolationMode.BILINEAR,
           'bicubic': InterpolationMode.BICUBIC,
           'lanczos': InterpolationMode.LANCZOS,
           'nearest': InterpolationMode.NEAREST}
def custom_resize(img, opt):
    interp = sample_discrete(opt.rz_interp)
    return TF.resize(img, (opt.loadSize,opt.loadSize), interpolation=rz_dict[interp])
