import torch,os
import pandas as pd
import numpy as np
from PIL import Image
import logging
import open_clip
import pickle

import cv2
from PIL import Image
import random
import numpy as np
import torch
import logging
from torch import distributed as dist, nn as nn
from torch.nn import functional as F
from scipy.optimize import fsolve


class DirectT:
    def __init__(self):
        pass
    def __call__(self,x,U=None):
        return x
    
class UniformBlur:
    def __init__(self,blur_kernel_size):
        self.blur_kernel_size = blur_kernel_size

    def __call__(self, img):
        if isinstance(img, torch.Tensor):
            img = F.to_pil_image(img)
        img_np = np.array(img)
        if img_np.shape[2] == 3:
            img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        img_blur = cv2.GaussianBlur(img_np, (self.blur_kernel_size, self.blur_kernel_size), 0)
        img_blur = cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB)
        blurred_image = Image.fromarray(img_blur)

        return Image.fromarray(img_blur)


class FoveaBlur:
    def __init__(self, h, w, blur_kernel_size, curve_type='exp', *args, **kwargs):
        self.blur_kernel_size = blur_kernel_size
        self.mask = np.zeros((h,w), np.float32)
        
        center = (w // 2, h // 2)
        max_distance = np.sqrt((h - center[1] - 1) ** 2 + (w - center[0] - 1) ** 2)
        c = 0.5
        center_resolution = 1-c
        edge_resolution = 0

        initial_guess = [1.0, 1.0]
        def equations(vars):
            t, r = vars
            eq1 = r * (t - np.sin(t)) - 1  # x = 1
            eq2 = -r * (1 - np.cos(t)) + 1.0  # y = 0
            return [eq1, eq2]
        solution = fsolve(equations, initial_guess)
        t_max, r_solution = solution
        self.r = r_solution

        fun_degrade = getattr(self, curve_type, None)
        for i in range(h):
            for j in range(w):
                distance = np.sqrt((i - center[1]) ** 2 + (j - center[0]) ** 2)
                x0 = min(1,distance/max_distance)
                y0 = fun_degrade(x0,**kwargs)
                self.mask[i, j] = edge_resolution + (center_resolution - edge_resolution) * y0

    def alphaBlend(self, img1, img2, mask):

        alpha = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        blended = cv2.convertScaleAbs(img1*(1-alpha) + img2*alpha)
        return blended
    
    def __call__(self, img, blur_kernel_size=None): 
        if blur_kernel_size ==None:
            blur_kernel_size = self.blur_kernel_size
        img = np.array(img)
        if img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        blured = cv2.GaussianBlur(img, (blur_kernel_size,blur_kernel_size), 0)
        blended = self.alphaBlend(img, blured, 1- self.mask)
        blended = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        return Image.fromarray(blended)
    
    def linear(self,x,**kwargs):
        return 1-x
    
    def exp(self,x,**kwargs):
        system_g = kwargs.get('system_g', 4)
        return  np.exp(-system_g * x)
    
    def quadratic(self,x,**kwargs):
        return  1 - x**2
    
    def log(self,x,**kwargs):
        b = 1/(np.e-1)
        a = np.log(b) + 1
        return  a - np.log(x + b)
    
    def brachistochrone(self,x,**kwargs):
        
        def equation(t):
            return t - np.sin(t) - (x / self.r)

        t0 = fsolve(equation, [1.0, 1.0])[0]
        y0 = -self.r * (1 - np.cos(t0)) + 1.0
        return  y0