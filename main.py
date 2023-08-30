import numpy as np
from PIL import Image

import kernels


class Kernel:
    def __init__(self, kernel):
        self.kernel = kernel
        self.height = self.h = len(kernel)
        self.width = self.w = len(kernel[0])


class Picture:
    def __init__(self, filename=None, rgb=None):
        if filename:
            pil_img = Image.open(filename)
            self.img = pil_img
            self.rgb = np.array(pil_img.getdata(), dtype=np.int8).reshape(pil_img.height, pil_img.width, 3)
        else:
            self.img = Image.fromarray(rgb)
            self.rgb = rgb
        self.r, self.g, self.b = self.rgb[:, :, 0], self.rgb[:, :, 1], self.rgb[:, :, 2]
        self.grayscale = 0.299*self.r + 0.587*self.g + 0.114*self.b
        self.height = self.h = len(self.rgb)
        self.width = self.w = len(self.rgb[0])

    @classmethod
    def from_rgb(cls, rgb):
        return cls(rgb=rgb)

    @staticmethod
    def save_img(img, filename):
        arr = np.asarray(img, dtype=np.uint8)
        pil_img = Image.fromarray(arr)
        pil_img.save(filename, format='png')

    @staticmethod
    def create_img(height, width, color):
        result = [None] * height
        for i in range(len(result)):
            result[i] = [color] * width
        return result

    @staticmethod
    def pad_array(arr, k_height, k_width, pad=0):
        return np.pad(arr, ((k_height//2,), (k_width//2,)), constant_values=pad)

    @staticmethod
    def get_grid(arr, x, y, height, width):
        return arr[x - height//2:x + height//2 + 1, y - width//2:y + width//2 + 1]

    def apply_kernel(self, arr, kernel):
        arr = self.pad_array(arr, kernel.h, kernel.w)
        new_arr = np.zeros_like(arr)
        for x in range(kernel.h//2, len(new_arr) - kernel.h//2):
            for y in range(kernel.w//2, len(new_arr[0]) - kernel.w//2):
                grid = self.get_grid(arr, x, y, kernel.h, kernel.w)
                new_arr[x, y] = np.sum(grid * kernel.kernel)
        return new_arr[kernel.h//2:-kernel.h//2, kernel.w//2:-kernel.w//2]

    def filter(self, kernel):
        r = self.apply_kernel(self.r, kernel)
        g = self.apply_kernel(self.g, kernel)
        b = self.apply_kernel(self.b, kernel)
        rgb = np.stack((r, g, b), axis=-1)
        return rgb


if __name__ == '__main__':
    pic = Picture('images/cat.jpg')
    test_kernel = Kernel(kernels.sharpen)
    blurred_arr = pic.filter(test_kernel)
    Picture.save_img(blurred_arr, 'DoesThisWork.png')
