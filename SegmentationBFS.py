from PIL import Image
from pix2tex.cli import LatexOCR
import pytesseract
import easyocr
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from collections import deque
import math
import pandas as pd
import os
from PIL import Image
from functools import cmp_to_key
import re


class Segmentation_BFS:
    

    def __init__(self, image_path=None, image=None):
        
        self.image_path = image_path
        self.image = image
        
        if not image is None:
            self.image_source = image.copy()
        else:
            self.image_source = None

    def get_image(self, image_path=None):
        
        if image_path is None:
            image_path = self.image_path
        
        img = cv2.imread(image_path,  cv2.IMREAD_GRAYSCALE)
        
        return img
    
    '''Cегментация'''
    
    #Ищет контур изображения
    def contour(self, image_path=None, image=None, tres=(235, 300), show=False):
        
        if not image_path is None:
            self.image_path = image_path
        
        if self.image is None: 
            img = cv2.imread(self.image_path,  cv2.IMREAD_GRAYSCALE)
            img = cv2.Canny(img, threshold1=tres[0], threshold2=tres[1])
        else:
            img = self.image
            img = cv2.Canny(img, threshold1=tres[0], threshold2=tres[1])
        self.image = img
        if show is True:
            plt.imshow(img, cmap='gray')

        return img

    #Размазывает одну строку/cтолбец (вектор) на precision в обе стороны
    def add_precision(self, vector, precision=1):
        if (precision == 0):
            return vector

        i = 0
        fin = len(vector)

        while True:
            if (vector[i] != 0):
                if ((i != 0) and (vector[i - 1] == 0)):
                    for j in range(max(0, i - precision), i):
                        vector[j] = vector[i]
                if ((i != len(vector) - 1) and (vector[i + 1] == 0)):
                    for j in range(i, min(len(vector), i + precision + 1)):
                        vector[j] = vector[i]
                    i = min(len(vector), i + precision + 1) - 1


            i += 1
            if i >= len(vector) - 1:
                break

        return vector
    
    
    # Получает на  вход 
    # Гиперпараметр pc (precision_coefficient), отвечающий за степень размытия вектора
    # Вектор нумпи, который предстает размыть
    # Размазывает хуету пропорционально ее размерам
    def MARAZM(self, vector, pc=0.5):

        copy = vector.copy()
        ln = len(vector)
        counter = 0

        i = 0
        while i <= ln - 1:
            if copy[i] != 0:
                counter += 1
                i += 1
            else:
                if counter != 0:
                    copy[max(0, math.ceil(i - counter - pc * counter)): min(ln - 1, math.floor(i + pc * counter))] = 1

                    i += math.floor(pc * counter)
                    i = min(ln, i)

                    counter = 0
                else:
                    i += 1

        return copy
    
    # Получает на  вход 
    # Гиперпараметр pc (precision_coefficient), отвечающий за степень размытия вектора
    # Вектор нумпи, который предстает размыть
    def MARAZM_root(self, vector, pc=0.5, ceiling=float('inf')):

        copy = vector.copy()

        ln = len(vector)
        counter = 0


        i = 0
        while i <= ln - 1:
            if copy[i] != 0:
                counter += 1
                i += 1
            else:
                if counter != 0:
                    copy[max(0, math.ceil(i - counter - pc * math.sqrt(counter)), i - counter - ceiling):\
                         min(ln - 1, math.floor(i + pc * math.sqrt(counter)), i + ceiling)] = 1

                    i += min(math.floor(pc * math.sqrt(counter)), ceiling)
                    i = min(ln, i)

                    counter = 0
                else:
                    i += 1

        return copy

    
    #Итеративно вызывает add_precision() или MARAZM() для каждой строки (если ось х) 
    #или каждого столбца (если ось у), тем самым размазывая изображение
    
    def blur(self, img=None, precision=0.5, ceiling=float('inf'), axis='x', kind = 'MARAZM', show=False):

        if img == None:
            t_img = self.image.copy()
        else:
            t_img = img.copy()

        if axis != 'x':
            t_img = t_img.T

        if kind == 'add_precision':
            for row in t_img:
                self.add_precision(row, precision)
        elif kind == 'MARAZM':
            for i in range(len(t_img)):
                t_img[i] = self.MARAZM(t_img[i], precision, ceiling=ceiling)
        elif kind == 'MARAZM_root':
            for i in range(len(t_img)):
                t_img[i] = self.MARAZM_root(t_img[i], precision, ceiling=ceiling)


        if axis != 'x':
            t_img = t_img.T

        if show is True:
            plt.imshow(t_img, cmap='gray')

        return t_img


    #Запускает обход в ширину при обнаружении размазни. Вызывается в следующей функции
    def BFS_segment(self, blurred, been, coords, show=False,
                   ret: str = 'segments'):

        segment = np.zeros_like(blurred)
        segment[coords[0]][coords[1]] = 1

        q_x = deque()
        q_x.append(coords[1])

        q_y = deque()
        q_y.append(coords[0])

        if ret == 'coords' or ret == 'both':
            min_x, max_x, min_y, max_y = float('inf'), 0, float('inf'), 0

        while q_x:
            i = q_y.popleft()
            j = q_x.popleft()

            if ret == 'coords' or ret == 'both':
                if j < min_x:
                    min_x = j
                if j > max_x:
                    max_x = j
                if i < min_y:
                    min_y = i
                if i > max_y:
                    max_y = i

            if (j - 1 >= 0) and (been[i][j - 1] == 0) and (blurred[i][j - 1] == 1):
                q_x.append(j - 1)
                q_y.append(i)

                been[i][j - 1] = 1
                segment[i][j - 1] = 1

            if (i - 1 >= 0) and (been[i - 1][j] == 0) and (blurred[i - 1][j] == 1):
                q_x.append(j)
                q_y.append(i - 1)

                been[i - 1][j] = 1
                segment[i - 1][j] = 1

            if (j + 1 < len(blurred[0])) and (been[i][j + 1] == 0) and (blurred[i][j + 1] == 1):
                q_x.append(j + 1)
                q_y.append(i)

                been[i][j + 1] = 1
                segment[i][j + 1] = 1

            if (i + 1 < len(blurred)) and (been[i + 1][j] == 0) and (blurred[i + 1][j] == 1):
                q_x.append(j)
                q_y.append(i + 1)

                been[i + 1][j] = 1
                segment[i + 1][j] = 1

        if show is True:
            plt.imshow(segment, cmap='gray')
            
        if ret == 'coords':
            return (min_x, max_x, min_y, max_y)
        elif ret == 'both':
            return (min_x, max_x, min_y, max_y), segment
        return segment

    #Ищет размазни, встречая их вызывает обход в ширину (BFS_segment)
    #ret = (segments, coords, both)
    def segmentation_BFS(self,
                        ret: str = 'segments'):
        
        blurred = self.image.copy()
        
        been = np.zeros_like(blurred)
        blurred[blurred != 0] = 1

        if ret == 'segments' or ret == 'both': 
            segments = []
        if ret == 'coords' or ret == 'both':
            coordinates = []

        while not np.array_equal(been, blurred):
            for i in range(len(blurred)):
                for j in range(len(blurred[i])):
                    if (blurred[i][j] != 0) and (been[i][j] == 0):
                        been[i][j] = 1
                        if ret == 'segments':
                            segment = self.BFS_segment(blurred, been, coords=(i, j), ret=ret, show=False)
                            segments.append(segment)
                        elif ret == 'coords':
                            coords = self.BFS_segment(blurred, been, coords=(i, j), ret=ret, show=False)
                            coordinates.append(coords)
                        elif ret == 'both':
                            coords, segment = self.BFS_segment(blurred, been, coords=(i, j), ret=ret, show=False)
                            segments.append(segment)
                            coordinates.append(coords)
        if ret == 'segments': 
            return segments
        elif ret == 'coords':
            return coordinates
        return coordinates, segments
       
       
    #Объединяет предыдущие функции для поиска сегментов
    def get_segments_old(self, image_path, cont_tres=(500, 600), precision=(6, 4)):
        
        print('it is a deprecated version\n')
        if not image_path is None:
            self.image_path = image_path
            
        cv2.imread(self.image_path,  cv2.IMREAD_GRAYSCALE)
        cont_img = self.contour(self.image_path, tres=cont_tres)
        blurred = self.blur(cont_img, precision=precision[0], axis='x')
        blurred = self.blur(blurred, precision=precision[1], axis='y')
        segments = self.segmentation_BFS(blurred)

        return segments
    
    def get_segments(self, image=None, image_path=None, cont_tres=(500, 600),
                     precision=(9, 2), ceiling=(10, 4), kind='MARAZM_root',
                     ret: str = 'segments', show: str = False):
        
        if not image_path is None:
            self.image_path = image_path
        if not image is None:
            self.image=image
            self.image_source=image.copy()
        
        self.image = self.contour(image_path=self.image_path, image=self.image, tres=cont_tres)
        self.image = self.blur(precision=1, kind='add_precision')
#         self.image = self.blur(precision=1, axis = 'y', kind='add_precision')
        segments = self.segmentation_BFS()
        for i in range(len(segments)):
            self.cut_projection(segment = segments[i], ret='fill')
        self.image = self.blur(precision=precision[0], ceiling=ceiling[0], kind=kind)
        self.image = self.blur(precision=precision[1], ceiling=ceiling[1], axis='y', kind=kind)
        
        
        if ret == 'segments':
            segments = self.segmentation_BFS(ret=ret)
            if show is True:
                plt.imshow(self.image, cmap='gray')
            return segments
        if ret == 'coords':
            coords = self.segmentation_BFS(ret=ret)
            return coords
        if ret == 'both':
            coords, segments = self.segmentation_BFS(ret=ret)
            if show is True:
                plt.imshow(self.image, cmap='gray')
            return coords, segments
    
    
    #returns array of cut image segments ('cv2' / 'PIL'), ret =('images', 'both')
    def get_array_of_image_segments(self, coords, image=None, image_path=None, ret='images',
                                   form: str = 'cv2'):
        
        array=[]
        if image is None:
            image = self.get_image(image_path)
        else:
            image = self.image_source.copy()
            
        for i in range(len(coords)):
            crd = coords[i]
            im = image[crd[2]: crd[3], crd[0]: crd[1]]
            if form == 'PIL':
                im=self.cv2_to_PIL(im)
            array.append(im)
        
        if ret == 'images':
            return array
        else:
            return coords, array
        
    
    #combines get_segments() and get_array_of_image_segments()
    def get_segments_as_images(self, image=None, image_path=None, cont_tres=(500, 600),
                               precision=(9, 2), ceiling=(10, 4), kind='MARAZM_root', form='cv2', ret='images'):
                  
        coords = self.get_segments(image, image_path, cont_tres, precision, ceiling, kind, ret='coords')
        
        if ret == 'images':
            array_image_segments = self.get_array_of_image_segments(coords, image, image_path, ret=ret, form=form)
            return array_image_segments
        elif ret == 'both':
            coords, array_image_segments = self.get_array_of_image_segments(coords, image, image_path, form=form, ret=ret)
            return coords, array_image_segments
        

    '''Методы визуализации и получения изображения'''
   
    #Возвращает кусок изображения принадлежащий сегменту
    def show_part_of_image(self, segment, image=None, show=False):
        
        if image is None:
            image = self.image
        mask = segment.copy()
        mask[mask != 0] = 1
        mask = mask.astype(bool)

        iz = np.where(mask.astype(bool), image, 0)
       
        if show is True:
            plt.imshow(iz, cmap='gray')

        return iz
   
    #Возвращает квадрат сегмента и сегмент в нем если ret == 'cut' (в виде маски) либо координаты сегмента
    # если ret == 'coords' или замазывает исходную хуету квадратиком если ret == 'fill'
    def cut_projection(self, segment, blurred=None, ret='cut', show=False):
        
        if blurred is None:
            blurred = self.image
        
        x_proj = segment.any(axis=0)
        y_proj = segment.any(axis=1)


        def find_first(proj):
            for i in range(len(proj)):
                if proj[i]!=0:
                    return i

        def find_last(proj):
            for i in range(len(proj) - 1, 0, -1):
                if proj[i]!=0:
                    return i

        y_first = find_first(y_proj)
        y_last = find_last(y_proj)
        x_first = find_first(x_proj)
        x_last = find_last(x_proj) 

        if ret == 'cut':
            cut_segment = segment.copy()[y_first:y_last, x_first:x_last]

            cut_segment[cut_segment!=0] = 255

            if show is True:
                plt.imshow(cut_segment, cmap='gray')

            return cut_segment

        elif ret == 'coords':
            if show is True:
                cut_segment = segment.copy()[y_first:y_last, x_first:x_last]

                cut_segment[cut_segment!=0] = 255

                plt.imshow(cut_segment, cmap='gray')

            return x_first, x_last, y_first, y_last

        elif ret == 'fill':
            blurred[y_first: y_last + 1, x_first:x_last + 1] = 1

            if show is True:
                blurcop = blurred.copy()
                blurcop[blurcop!=0] = 255
                plt.imshow(blurcop, cmap='gray')

        else:
            return None

    
    '''cv2 -> PIL'''
    #Перевод cv2 -> PIL
    def cv2_to_PIL(self, opencv_image):

        color_converted = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
        pil_image=Image.fromarray(color_converted)

        return pil_image
    

