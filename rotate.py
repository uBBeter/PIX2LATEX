import cv2
import numpy as np
from PIL import Image

def rotate_image(image):
    
    # Преобразуем изображение в оттенки серого
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Применяем размытие для уменьшения шума
    blur = gray # cv2.GaussianBlur(gray, (9, 9), 0)
    # Оставил нулевым т.к. для текста и так норм
    
    # Используем детектор границ
    edges = cv2.Canny(blur, 50, 150, apertureSize=3)
    
    # Ищем линии на изображении методом Хафа
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)
    
    # Подсчитываем средний угол наклона линий
    if lines is not None:
        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            angles.append(angle)
        
        # Вычисляем медиану углов
        median_angle = np.median(angles)
        
        # Поворачиваем изображение для выравнивания
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
        rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        
        return Image.fromarray(rotated_image)
    else:
        return Image.fromarray(image)  # Возвращаем исходное изображение, если линии не найдены
