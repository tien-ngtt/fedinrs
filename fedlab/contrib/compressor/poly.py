import math
import torch
import sympy as sp
import numpy as np

from .compressor import Compressor
from scipy.interpolate import lagrange


class PolyCompressor(Compressor):
    
    def __init__(self, degree: int = 7, interval: int = 100):
        "Degree is integer number of expansion in Taylor series, interval is partition of weights into smaller segments"
        self.degree = degree if degree >= 2 else 2
        self.interval = interval if interval > 0 else 100

    def compress(self, weights):
        
        if torch.is_tensor(weights):
            weights = weights.detach()
        else:
            raise TypeError(
                "Invalid type error, expecting {}, but get {}".format(
                    torch.Tensor, type(tensor)))

        num_weights = weights.numel()
        enum = [i for i in range(num_weights)]

        weights = weights.view(-1)
        weights = weights.numpy()
        x_sym = sp.symbols('x')
        poly_coeffs = []
        remaining = num_weights
        iterator = 0 
        
        while (remaining > 0):
            minimum = min(self.interval, remaining)
            enum = np.array([i for i in range(minimum)])
            lagr = lagrange(enum, weights[iterator:(iterator + minimum)])
            coef = lagr.c
            coef_length = len(coef)
            powers = np.arange(coef_length)[::-1]
            lagr_sym = sum([i * x_sym** j for i, j in zip(coef,powers)])
            series = sp.series(lagr_sym, x_sym, int(np.argmax(abs(weights[iterator:(iterator + minimum)]))), self.degree)
            approx = series.removeO()
            poly = sp.expand(approx)
            if poly.as_poly() is None:
                coeffs = torch.tensor([poly],dtype=torch.float32)            
            else:
                coefficients = [poly.coeff(x_sym, n) for n in range(poly.as_poly().degree() + 1)]
                g = []
                for i in range(minimum):
                    g.append(sum([m*(i**n) for m,n in zip(coefficients, range(len(coefficients)))]))
                count = []
                diff = [abs((k - l)/l) for k,l in zip(g,weights[iterator:(iterator + minimum)])]
                for k in range(len(diff)):
                    if (diff[k] <= 0.1):
                        count.append(k)
                first_element = min(count) if count != [] else 0
                coeffs = [torch.tensor(coefficients,dtype=torch.float32), torch.tensor(first_element,dtype=torch.int8), torch.tensor(len(count),dtype=torch.int8)]
                
            poly_coeffs.append(coeffs)
            
            iterator = iterator + minimum
            remaining = remaining - minimum
        
        return poly_coeffs

    def decompress(self, poly_coeffs, num_elements, shape):
        remain = num_elements
        iterate= 0
        de_weights = torch.tensor([])
        while (remain > 0):
            mi = min(remain, self.interval)
            de_enum = torch.tensor([i for i in range(mi)])
            poly_c = poly_coeffs[iterate]
            if (len(poly_c) > 1):
                if (poly_c[2] == 0):
                    for i in de_enum:
                        de_weights = torch.cat((de_weights,torch.tensor(0).unsqueeze(0)),dim=0)
                else:
                    for i in de_enum:
                        de_element = torch.tensor(0).unsqueeze(0) if ((i<poly_c[1]) or (i>(poly_c[1]+poly_c[2]-1))) else torch.sum(torch.tensor([j*(i**k) for j,k in zip(poly_c[0],range(len(poly_c[0])))])).unsqueeze(0)
                        de_weights = torch.cat((de_weights,de_element),dim=0)
                               
            else:
                for i in de_enum:
                    de_element = torch.sum(torch.tensor([j*(i**k) for j,k in zip(poly_c,range(len(poly_c)))])).unsqueeze(0) 
                    de_weights = torch.cat((de_weights,de_element),dim=0)
                
            remain = remain - mi
            iterate += 1
            
        de_weights = de_weights.view(shape)
        
        return de_weights