"""
5h) (5 točk) Pridobljeno znanje o uporabi morfoloških operacij preizkusite še na bolj
realnem primeru. Preberite sliko bird.jpg, spremenite jo v sivinsko ter določite prag,
da dobite čim boljšo masko objekta. Ker popolne maske ne morete dobiti samo z
globalnim pragom, jo izboljšajte z uporabo morfoloških operacij. Število točk, ki jih
boste dobili za nalogo je odvisno od kakovosti rezultata.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')

im = cv2.imread('bird.jpg')
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
mask_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=11)

plt.figure(figsize=(12, 5))
plt.subplot(1, 3, 1)
plt.imshow(gray, cmap='gray')
plt.title('5h) Sivinska slika')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(mask, cmap='gray')
plt.title('5h) Otsu maska')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(mask_close, cmap='gray')
plt.title('5h) Morfološka maska')
plt.axis('off')

plt.tight_layout()
plt.show()
