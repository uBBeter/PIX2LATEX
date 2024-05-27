from PIL import Image
from pix2tex.cli import LatexOCR
import pytesseract
import numpy as np
import cv2
from collections import deque
import pandas as pd
from PIL import Image
from functools import cmp_to_key
import re
from SegmentationBFS import Segmentation_BFS
import rotate



class getTex:
    def __init__(self, img=None, type="jpg"):
        self.img = img
        self.type = type





    def process_image(self, img):
        # img = rotate.rotate_image(img)
        df = self.get_df(img)
        text_results = self.getTextResults(df, img)
        img = self.onlyMath(img, text_results)
        math_results = self.getMathResults(img)
        return self.makeText(text_results, math_results)





    def get_df(self, img):
        pytesseract.pytesseract.tesseract_cmd = 'YOUR_PATH_TO_TESSERACT'
        df = pytesseract.image_to_data(img, output_type='data.frame', lang="rus")
        return df
    


    
    def getTextResults(self, df, img):
        first_text_results = []
        alphabet = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя,.'
        words_df = df[df['conf'] > 90]
        first_text_results = list(words_df[['line_num', 'text', 'left', 'top', 'width', 'height']].itertuples(index=False, name=None))
        for i in range(len(first_text_results)):
            first_text_results[i] = list(first_text_results[i]) + [False]


        text_results = []
        for word in first_text_results:
            if not re.sub('[`:.,]', '', word[1]).isdigit() and len(word[1]) > 1 or len(word[1]) == 1 and word[1].lower() in alphabet:
                text_results.append(word)
        return text_results
    


    def onlyMath(self, img, text_results):
        pixels = img.load()
        fill_color = pixels[1, 1]

        for word in text_results:
            left = word[2]
            top = word[3]
            width = word[4]
            height = word[5]
            for x in range(left, left + width):
                for y in range(top, top + height):
                    pixels[x, y] = fill_color

        return img
    

    def getMathResults(self, img):
        segm = Segmentation_BFS()
        img = cv2.cvtColor(np.array(img), cv2.THRESH_BINARY)
        coords, vector = segm.get_segments_as_images(image=img, form="PIL", ret="both", precision=(3, 3), ceiling=(float("inf"), float("inf")))
        img = Image.fromarray(img)


        model = LatexOCR()
        math_results = []

        thresh = 210
        fn = lambda x : 255 if x > thresh else 0
        img = img.convert('L').point(fn, mode='1')

        for segment in coords:
            expression = model(img.convert('1').crop((segment[0], segment[2], segment[1], segment[3])))
            math_results.append([0, expression,
                                segment[0], segment[2], segment[1] - segment[0], segment[3] - segment[2], True])
            
        return math_results
    
    def makeText(self, text_results, math_results):
        results = math_results + text_results
        results = [[x[6], (x[2], x[3], x[4], x[5]), x[1], x[0]] for x in results]


        def calculate_center(coords):
            x1, y1, width, height = coords
            center_x = (x1 + x1 + width) // 2
            center_y = (y1 + y1 + height) // 2
            return center_x, center_y

        def custom_comparator(element1, element2):
            _, coords1, _, _= element1
            _, coords2, _, _= element2
            
            center_x1, center_y1 = calculate_center(coords1)
            center_x2, center_y2 = calculate_center(coords2)
            
            if abs(center_y1 - center_y2) < 15:
                if center_x1 < center_x2:
                    return -1
                elif center_x1 > center_x2:
                    return 1
                else:
                    return 0
            else:
                if center_y1 < center_y2:
                    return -1
                else:
                    return 1

        sorted_elements = sorted(results, key=cmp_to_key(custom_comparator))


        string_counter = 1
        line_array = [1]
        for i in range(1, len(sorted_elements) - 1):
            if  sorted_elements[i][-1] == 0:
                if sorted_elements[i - 1][-1] > sorted_elements[i][-1]:
                    _, coords1, _, _= sorted_elements[i - 1]
                    _, coords2, _, _= sorted_elements[i]
                    center_x1, center_y1 = calculate_center(coords1)
                    center_x2, center_y2 = calculate_center(coords2)
                    if abs(center_y2 - center_y1) > 10:
                        string_counter += 1
            else:
                if sorted_elements[i - 1][-1] != sorted_elements[i][-1]:
                    string_counter += 1
            line_array.append(string_counter)
        line_array.append(line_array[-1])

        for i in range(len(sorted_elements)):
            sorted_elements[i][-1] = line_array[i]

        begin = "\\documentclass{article}\n\\usepackage{amsmath}\n\\usepackage[T2A]{fontenc}\n\\usepackage[russian, english]{babel}\n\\begin{document}\n"
        end = "\n\\end{document}"
        tex = begin + sorted_elements[0][2] + " "
        for i in range(1, len(sorted_elements)):
            if sorted_elements[i][0] == True:
                if sorted_elements[i - 1][-1] < sorted_elements[i][-1]:
                    tex += '\n\[\n' + sorted_elements[i][2] + "\n\]\n"
                    continue
                else:
                    tex += '\(' + sorted_elements[i][2] + "\) "
                    continue
            else:
                if sorted_elements[i - 1][-1] < sorted_elements[i][-1]:
                    tex += "\\\\"
            tex += sorted_elements[i][2] + " "
        tex += end
        return tex




# image_path = "example.jpg"
# img = Image.open(image_path)
# print(getTex().process_image(img))


