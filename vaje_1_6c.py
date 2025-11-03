"""
6c) (10 točk) Izberite si predmet poljubne barve in z uporabo znanja iz prejšnjih nalog upragujte sliko s takšnimi vrednostmi,
da boste iz nje izluščili izbrani predmet. Z uporabo OpenCV funkcije cv2.findContours() izrišite konturo okoli izluščenih predmetov.
Število točk, ki jih boste dobili za to nalogo bo odvisno od robustnosti delovanja vaše metode.

prepozna rumeno barvo - rgb(255, 251, 0)
"""
import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 120, 120])
    upper = np.array([150, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)
    cv2.imshow('6c) Mask', mask)
    cv2.imshow('6c) Frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
