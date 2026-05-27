from __future__ import division
import torch
import math
import random
from PIL import Image, ImageOps, ImageEnhance
try:
    import accimage
except ImportError:
    accimage = None
import numpy as np
import scipy.stats as st
import cv2
import numbers
import types
import collections
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import util.util as util
from scipy.signal import convolve2d


# utility
def _is_pil_image(img):
    if accimage is not None:
        return isinstance(img, (Image.Image, accimage.Image))
    else:
        return isinstance(img, Image.Image)


def _is_tensor_image(img):
    return torch.is_tensor(img) and img.ndimension() == 3


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def arrshow(arr):
    Image.fromarray(arr.astype(np.uint8)).show()


def get_transform(opt):
    transform_list = []
    osizes = util.parse_args(opt.loadSize)
    fineSize = util.parse_args(opt.fineSize)
    if opt.resize_or_crop == 'resize_and_crop':    
        transform_list.append(
            transforms.RandomChoice([
                transforms.Resize([osize, osize], Image.BICUBIC) for osize in osizes
            ]))
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    return transforms.Compose(transform_list)


to_norm_tensor = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        (0.5, 0.5, 0.5),
        (0.5, 0.5, 0.5)
    )
])

to_tensor = transforms.ToTensor()


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    h = math.ceil(h / 2.) * 2  # round up to even
    return img.resize((w, h), Image.BICUBIC)


# functional 
def gaussian_blur(img, kernel_size, sigma):
    from scipy.ndimage.filters import gaussian_filter
    if not _is_pil_image(img):
        raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

    img = np.asarray(img)
    # the 3rd dimension (i.e. inter-band) would be filtered which is unwanted for our purpose
    # new = gaussian_filter(img, sigma=sigma, truncate=truncate)
    if isinstance(kernel_size, int):
        kernel_size = (kernel_size, kernel_size)
    elif isinstance(kernel_size, collections.Sequence):
        assert len(kernel_size) == 2        
    new = cv2.GaussianBlur(img, kernel_size, sigma)  # apply gaussian filter band by band    
    return Image.fromarray(new)


# transforms
class GaussianBlur(object):
    def __init__(self, kernel_size=11, sigma=3):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        return gaussian_blur(img, self.kernel_size, self.sigma)


class ReflectionSythesis_1(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of ICCV 2017 paper *"A Generic Deep Architecture for Single Image Reflection Removal and Image Smoothing"*    
    """
    def __init__(self, kernel_sizes=None, low_sigma=2, high_sigma=5, low_gamma=1.3, high_gamma=1.3):
        self.kernel_sizes = kernel_sizes or [11]
        self.low_sigma = low_sigma
        self.high_sigma = high_sigma
        self.low_gamma = low_gamma
        self.high_gamma = high_gamma
        print('[i] reflection sythesis model: {}'.format({
            'kernel_sizes': kernel_sizes, 'low_sigma': low_sigma, 'high_sigma': high_sigma,
            'low_gamma': low_gamma, 'high_gamma': high_gamma}))

    def __call__(self, B, R):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))
        
        B_ = np.asarray(B, np.float32) / 255.
        R_ = np.asarray(R, np.float32) / 255.

        kernel_size = np.random.choice(self.kernel_sizes)
        sigma = np.random.uniform(self.low_sigma, self.high_sigma)
        gamma = np.random.uniform(self.low_gamma, self.high_gamma)
        R_blur = R_
        kernel = cv2.getGaussianKernel(11, sigma)
        kernel2d = np.dot(kernel, kernel.T)

        for i in range(3):
            R_blur[...,i] = convolve2d(R_blur[...,i], kernel2d, mode='same')

        M_ = B_ + R_blur
        
        if np.max(M_) > 1:
            m = M_[M_ > 1]
            m = (np.mean(m) - 1) * gamma
            R_blur = np.clip(R_blur - m, 0, 1)
            M_ = np.clip(R_blur + B_, 0, 1)
        
        return B_, R_blur, M_


def _float_image(image):
    return np.asarray(image, np.float32) / 255.0


def _normalize01(image):
    image = np.asarray(image, np.float32)
    min_value = float(image.min())
    max_value = float(image.max())
    if max_value - min_value < 1e-6:
        return np.zeros_like(image)
    return (image - min_value) / (max_value - min_value)


def _shift_float_image(image, dx, dy):
    height, width = image.shape[:2]
    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


class AdvancedReflectionSythesis(object):
    """Harder online reflection synthesis used by the training dataset.

    I = tau * B + alpha * M(x,y) * Blur(A(R)) + glare + noise
    """

    def __init__(
        self,
        alpha_range=(0.08, 0.35),
        tau_range=(0.82, 1.00),
        ghost_prob=0.25,
        glare_prob=0.20,
        structured_mask_prob=0.25,
        sharp_prob=0.10,
        noise_sigma_range=(0.0, 0.008),
    ):
        self.alpha_range = alpha_range
        self.tau_range = tau_range
        self.ghost_prob = ghost_prob
        self.glare_prob = glare_prob
        self.structured_mask_prob = structured_mask_prob
        self.sharp_prob = sharp_prob
        self.noise_sigma_range = noise_sigma_range
        self.last_params = {}
        self.last_mask = None
        self.last_glare = None
        print('[i] reflection sythesis model: advanced {}'.format({
            'alpha_range': alpha_range,
            'tau_range': tau_range,
            'ghost_prob': ghost_prob,
            'glare_prob': glare_prob,
            'structured_mask_prob': structured_mask_prob,
            'sharp_prob': sharp_prob,
            'noise_sigma_range': noise_sigma_range,
        }))

    def _color_and_exposure(self, reflection):
        temperature = np.random.uniform(-0.25, 0.25)
        exposure = np.random.uniform(0.65, 1.35)
        gamma = np.random.uniform(0.75, 1.25)

        gains = np.array([1.0 + temperature, 1.0, 1.0 - temperature], dtype=np.float32)
        reflection = np.clip(reflection * gains.reshape(1, 1, 3), 0, 1)
        reflection = np.clip(reflection * exposure, 0, 1)
        reflection = np.power(reflection, gamma)
        return reflection, {
            'temperature': float(temperature),
            'exposure': float(exposure),
            'reflection_gamma': float(gamma),
        }

    def _warp_reflection(self, reflection):
        height, width = reflection.shape[:2]

        angle = np.random.uniform(-3, 3)
        scale = np.random.uniform(0.97, 1.03)
        tx = np.random.uniform(-0.025, 0.025) * width
        ty = np.random.uniform(-0.025, 0.025) * height
        affine = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        affine[:, 2] += [tx, ty]
        warped = cv2.warpAffine(
            reflection,
            affine,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )

        max_shift = np.random.uniform(0.0, 0.025)
        src = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        jitter = np.float32([
            [np.random.uniform(0, max_shift) * width, np.random.uniform(0, max_shift) * height],
            [width - 1 - np.random.uniform(0, max_shift) * width, np.random.uniform(0, max_shift) * height],
            [width - 1 - np.random.uniform(0, max_shift) * width, height - 1 - np.random.uniform(0, max_shift) * height],
            [np.random.uniform(0, max_shift) * width, height - 1 - np.random.uniform(0, max_shift) * height],
        ])
        matrix = cv2.getPerspectiveTransform(src, jitter)
        warped = cv2.warpPerspective(
            warped,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT_101,
        )
        return warped, {
            'affine_angle': float(angle),
            'affine_scale': float(scale),
            'affine_tx': float(tx),
            'affine_ty': float(ty),
            'perspective_max_shift': float(max_shift),
        }

    def _gaussian_blur_reflection(self, reflection):
        kernel = int(np.random.choice([7, 9, 11, 15]))
        sigma = np.random.uniform(1.2, 4.5)
        return cv2.GaussianBlur(reflection, (kernel, kernel), sigma), {
            'blur': 'gaussian',
            'blur_kernel': kernel,
            'blur_sigma': float(sigma),
        }

    def _motion_blur_reflection(self, reflection):
        kernel_size = int(np.random.choice([5, 7, 9, 11, 15]))
        angle = np.random.uniform(0, 180)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        kernel[kernel_size // 2, :] = 1.0
        rotation = cv2.getRotationMatrix2D((kernel_size / 2.0 - 0.5, kernel_size / 2.0 - 0.5), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rotation, (kernel_size, kernel_size))
        kernel = kernel / max(kernel.sum(), 1e-6)
        return cv2.filter2D(reflection, -1, kernel), {
            'blur': 'motion',
            'blur_kernel': kernel_size,
            'motion_angle': float(angle),
        }

    def _defocus_blur_reflection(self, reflection):
        radius = int(np.random.choice([1, 2, 3, 4, 5]))
        kernel_size = radius * 2 + 1
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
        kernel = kernel / max(kernel.sum(), 1e-6)
        return cv2.filter2D(reflection, -1, kernel), {
            'blur': 'defocus',
            'defocus_radius': radius,
        }

    def _blur_reflection(self, reflection):
        if random.random() < self.sharp_prob:
            return reflection, {
                'blur': 'sharp',
                'blur_applied': False,
                'sharp_mix': 1.0,
            }

        blur_fn = random.choice([
            self._gaussian_blur_reflection,
            self._motion_blur_reflection,
            self._defocus_blur_reflection,
        ])
        blurred, params = blur_fn(reflection)
        if random.random() < 0.8:
            sharp_mix = np.random.uniform(0.35, 0.85)
            mixed = np.clip(sharp_mix * reflection + (1.0 - sharp_mix) * blurred, 0, 1)
        else:
            sharp_mix = 0.0
            mixed = blurred
        params.update({
            'blur_applied': True,
            'sharp_mix': float(sharp_mix),
        })
        return mixed, params

    def _add_ghosting(self, reflection):
        if random.random() >= self.ghost_prob:
            return reflection, {
                'ghost_enabled': False,
                'ghost_beta': 0.0,
                'ghost_shift': [0, 0],
            }

        beta = np.random.uniform(0.08, 0.22)
        max_shift = max(2, int(min(reflection.shape[:2]) * 0.025))
        dx = np.random.randint(-max_shift, max_shift + 1)
        dy = np.random.randint(-max_shift, max_shift + 1)
        if dx == 0 and dy == 0:
            dx = max_shift
        ghost = np.clip(reflection + beta * _shift_float_image(reflection, dx, dy), 0, 1)
        return ghost, {
            'ghost_enabled': True,
            'ghost_beta': float(beta),
            'ghost_shift': [int(dx), int(dy)],
        }

    def _low_frequency_noise(self, height, width):
        small_h = max(4, height // 32)
        small_w = max(4, width // 32)
        noise = np.random.rand(small_h, small_w).astype(np.float32)
        noise = cv2.resize(noise, (width, height), interpolation=cv2.INTER_CUBIC)
        noise = cv2.GaussianBlur(noise, (0, 0), sigmaX=max(width, height) * 0.04)
        return _normalize01(noise)

    def _structured_streak_mask(self, height, width):
        mask = np.zeros((height, width), dtype=np.float32)
        orientation = random.choice(['vertical', 'diagonal'])
        count = np.random.randint(1, 5)

        for _ in range(count):
            strength = np.random.uniform(0.25, 0.8)
            thickness = int(np.random.uniform(0.035, 0.12) * min(height, width))
            if orientation == 'vertical':
                x0 = int(np.random.uniform(-0.1, 1.1) * width)
                y0 = -height
                x1 = int(x0 + np.random.uniform(-0.15, 0.15) * width)
                y1 = height * 2
            else:
                start_left = random.random() < 0.5
                x0 = -width if start_left else width * 2
                y0 = int(np.random.uniform(-0.2, 1.0) * height)
                x1 = width * 2 if start_left else -width
                y1 = int(y0 + np.random.uniform(-0.6, 0.6) * height)
            cv2.line(mask, (int(x0), int(y0)), (int(x1), int(y1)), float(strength), thickness, cv2.LINE_AA)

        sigma = np.random.uniform(8.0, 28.0)
        mask = cv2.GaussianBlur(mask, (0, 0), sigma)
        return _normalize01(mask), {
            'structured_mask_enabled': True,
            'structured_mask_orientation': orientation,
            'structured_mask_count': int(count),
            'structured_mask_sigma': float(sigma),
        }

    def _spatial_mask(self, height, width):
        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        cx = np.random.uniform(0.25, 0.75) * width
        cy = np.random.uniform(0.25, 0.75) * height
        dist = np.sqrt(((x - cx) / max(width, 1)) ** 2 + ((y - cy) / max(height, 1)) ** 2)
        radial = _normalize01(dist)
        mode = random.choice(['center', 'edge'])
        if mode == 'center':
            radial = 1.0 - radial

        noise = self._low_frequency_noise(height, width)
        mix = np.random.uniform(0.35, 0.75)
        mask = _normalize01(mix * radial + (1.0 - mix) * noise)
        structured_params = {
            'structured_mask_enabled': False,
            'structured_mask_orientation': 'none',
            'structured_mask_count': 0,
        }
        if random.random() < self.structured_mask_prob:
            streak_mask, structured_params = self._structured_streak_mask(height, width)
            streak_strength = np.random.uniform(0.25, 0.65)
            mask = _normalize01((1.0 - streak_strength) * mask + streak_strength * streak_mask)
            structured_params['structured_mask_strength'] = float(streak_strength)

        mask = np.clip(0.55 + 0.45 * mask, 0, 1)
        params = {
            'mask_mode': mode,
            'mask_mix_radial': float(mix),
            'mask_center': [float(cx), float(cy)],
        }
        params.update(structured_params)
        return mask[..., None], params

    def _add_glare(self, height, width):
        glare = np.zeros((height, width, 3), dtype=np.float32)
        if random.random() >= self.glare_prob:
            return glare, {
                'glare_enabled': False,
                'glare_soft_spots': 0,
                'glare_ellipse_spots': 0,
                'glare_streaks': 0,
            }

        y, x = np.mgrid[0:height, 0:width].astype(np.float32)
        soft_spot_count = np.random.randint(1, 4)
        for _ in range(soft_spot_count):
            cx = np.random.uniform(0, width)
            cy = np.random.uniform(0, height)
            sigma = np.random.uniform(0.03, 0.16) * max(height, width)
            strength = np.random.uniform(0.08, 0.3)
            color = np.array([
                np.random.uniform(0.9, 1.0),
                np.random.uniform(0.82, 1.0),
                np.random.uniform(0.65, 1.0),
            ], dtype=np.float32)
            spot = np.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * sigma * sigma))
            glare += strength * spot[..., None] * color.reshape(1, 1, 3)

        ellipse_count = np.random.randint(1, 5)
        for _ in range(ellipse_count):
            overlay = np.zeros_like(glare)
            center = (int(np.random.uniform(0, width)), int(np.random.uniform(0, height)))
            axes = (
                int(np.random.uniform(0.05, 0.22) * width),
                int(np.random.uniform(0.015, 0.08) * height),
            )
            angle = float(np.random.uniform(-35, 35))
            strength = float(np.random.uniform(0.08, 0.32))
            color = (
                strength,
                strength * np.random.uniform(0.92, 1.0),
                strength * np.random.uniform(0.82, 1.0),
            )
            cv2.ellipse(overlay, center, axes, angle, 0, 360, color, -1, cv2.LINE_AA)
            glare += cv2.GaussianBlur(overlay, (0, 0), np.random.uniform(2.0, 8.0))

        streak_count = np.random.randint(2, 6)
        for _ in range(streak_count):
            overlay = np.zeros_like(glare)
            x0 = int(np.random.uniform(-0.2, 1.0) * width)
            y0 = int(np.random.uniform(0, height))
            length = int(np.random.uniform(0.35, 1.2) * width)
            angle = np.random.uniform(-35, 35) * np.pi / 180.0
            x1 = int(x0 + length * np.cos(angle))
            y1 = int(y0 + length * np.sin(angle))
            strength = float(np.random.uniform(0.08, 0.32))
            thickness = int(np.random.choice([1, 2, 3, 4]))
            color = (
                strength,
                strength * np.random.uniform(0.92, 1.0),
                strength * np.random.uniform(0.82, 1.0),
            )
            cv2.line(overlay, (x0, y0), (x1, y1), color, thickness, cv2.LINE_AA)
            blur_sigma = np.random.uniform(2.0, 10.0)
            glare += cv2.GaussianBlur(overlay, (0, 0), blur_sigma)

        glare = np.clip(glare, 0, 1)
        return glare, {
            'glare_enabled': True,
            'glare_soft_spots': int(soft_spot_count),
            'glare_ellipse_spots': int(ellipse_count),
            'glare_streaks': int(streak_count),
        }

    def __call__(self, B, R, return_extras=False):
        if not _is_pil_image(B):
            raise TypeError('B should be PIL Image. Got {}'.format(type(B)))
        if not _is_pil_image(R):
            raise TypeError('R should be PIL Image. Got {}'.format(type(R)))

        background = _float_image(B)
        reflection = _float_image(R)
        alpha = np.random.uniform(*self.alpha_range)
        tau = np.random.uniform(*self.tau_range)

        reflection, color_params = self._color_and_exposure(reflection)
        reflection, warp_params = self._warp_reflection(reflection)
        reflection, blur_params = self._blur_reflection(reflection)
        reflection, ghost_params = self._add_ghosting(reflection)

        height, width = background.shape[:2]
        mask, mask_params = self._spatial_mask(height, width)
        glare, glare_params = self._add_glare(height, width)
        noise_sigma = np.random.uniform(*self.noise_sigma_range)
        noise = np.random.normal(0.0, noise_sigma, background.shape).astype(np.float32)

        reflection_layer = np.clip(mask * reflection, 0, 1).astype(np.float32)
        blended = tau * background + alpha * reflection_layer + glare + noise
        blended = np.clip(blended, 0, 1).astype(np.float32)
        background = background.astype(np.float32)

        params = {
            'alpha': float(alpha),
            'tau': float(tau),
            'noise_sigma': float(noise_sigma),
        }
        params.update(color_params)
        params.update(warp_params)
        params.update(blur_params)
        params.update(ghost_params)
        params.update(mask_params)
        params.update(glare_params)

        self.last_params = params
        self.last_mask = mask.astype(np.float32)
        self.last_glare = glare.astype(np.float32)
        if return_extras:
            return background, reflection_layer, blended, self.last_mask, self.last_glare, params
        return background, reflection_layer, blended


class MixedReflectionSythesis(object):
    def __init__(self, reflection2_ratio=0.8, advanced_ratio=0.2):
        self.reflection2 = ReflectionSythesis_2()
        self.advanced = AdvancedReflectionSythesis()
        self.last_choice = None
        self.set_ratios(reflection2_ratio, advanced_ratio)

    def set_ratios(self, reflection2_ratio, advanced_ratio):
        total = float(reflection2_ratio + advanced_ratio)
        if total <= 0:
            raise ValueError('At least one synthesis ratio must be positive.')
        self.reflection2_ratio = float(reflection2_ratio) / total
        self.advanced_ratio = float(advanced_ratio) / total

    def __call__(self, B, R):
        if random.random() < self.reflection2_ratio:
            self.last_choice = 'reflection2'
            return self.reflection2(B, R)

        self.last_choice = 'advanced'
        return self.advanced(B, R)


class Sobel(object):
    def __call__(self, img):
        if not _is_pil_image(img):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        gray_img = np.array(img.convert('L'))
        x = cv2.Sobel(gray_img,cv2.CV_16S,1,0)
        y = cv2.Sobel(gray_img,cv2.CV_16S,0,1)
        
        absX = cv2.convertScaleAbs(x)   
        absY = cv2.convertScaleAbs(y)
        
        dst = cv2.addWeighted(absX,0.5,absY,0.5,0)
        return Image.fromarray(dst)


class ReflectionSythesis_2(object):
    """Reflection image data synthesis for weakly-supervised learning 
    of CVPR 2018 paper *"Single Image Reflection Separation with Perceptual Losses"*
    """
    def __init__(self, kernel_sizes=None):
        self.kernel_sizes = kernel_sizes or np.linspace(1,5,80)
    
    @staticmethod
    def gkern(kernlen=100, nsig=1):
        """Returns a 2D Gaussian kernel array."""
        interval = (2*nsig+1.)/(kernlen)
        x = np.linspace(-nsig-interval/2., nsig+interval/2., kernlen+1)
        kern1d = np.diff(st.norm.cdf(x))
        kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
        kernel = kernel_raw/kernel_raw.sum()
        kernel = kernel/kernel.max()
        return kernel

    def __call__(self, t, r):        
        t = np.float32(t) / 255.
        r = np.float32(r) / 255.
        ori_t = t
        # create a vignetting mask
        g_mask=self.gkern(560,3)
        g_mask=np.dstack((g_mask,g_mask,g_mask))
        sigma=self.kernel_sizes[np.random.randint(0, len(self.kernel_sizes))]

        t=np.power(t,2.2)
        r=np.power(r,2.2)
        
        sz=int(2*np.ceil(2*sigma)+1)
        
        r_blur=cv2.GaussianBlur(r,(sz,sz),sigma,sigma,0)
        blend=r_blur+t
        
        att=1.08+np.random.random()/10.0
        
        for i in range(3):
            maski=blend[:,:,i]>1
            mean_i=max(1.,np.sum(blend[:,:,i]*maski)/(maski.sum()+1e-6))
            r_blur[:,:,i]=r_blur[:,:,i]-(mean_i-1)*att
        r_blur[r_blur>=1]=1
        r_blur[r_blur<=0]=0

        h,w=r_blur.shape[0:2]
        neww=np.random.randint(0, 560-w-10)
        newh=np.random.randint(0, 560-h-10)
        alpha1=g_mask[newh:newh+h,neww:neww+w,:]
        alpha2 = 1-np.random.random()/5.0
        r_blur_mask=np.multiply(r_blur,alpha1)
        blend=r_blur_mask+t*alpha2
        
        t=np.power(t,1/2.2)
        r_blur_mask=np.power(r_blur_mask,1/2.2)
        blend=np.power(blend,1/2.2)
        blend[blend>=1]=1
        blend[blend<=0]=0
        
        return np.float32(ori_t), np.float32(r_blur_mask), np.float32(blend)


# Examples
if __name__ == '__main__':
    """cv2 imread"""
    # img = cv2.imread('testdata_reflection_real/19-input.png')
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = cv2.GaussianBlur(img, (11,11), 3)    

    """Sobel Operator"""
    # img = np.array(Image.open('datasets/VOC224/train/B/2007_000250.png').convert('L'))


    """Reflection Sythesis"""
    b = Image.open('datasets/VOCsmall/train/B/2008_000148.png')
    r = Image.open('datasets/VOCsmall/train/B/2007_000243.png')
    G = ReflectionSythesis_1()
    m, r = G(b, r)
    r.show()
    
    # img2 = gaussian_blur(img, 11, 3)
    # img2 = GaussianBlur(1, 1)(img)
    # print(np.sum(np.array(img2) - np.array(img)))
    # img2.show()
