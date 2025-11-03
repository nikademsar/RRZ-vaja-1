"""
2e) (10 točk) Za naslednjo nalogo boste potrebovali spletno kamero ter knjižnico OpenCV.
Spletna kamera sicer ni čista kamera z luknjico, vsebuje lečo, zato pri zajemu slike prihaja do določene stopnje popačenja.
Kljub temu z uporabo kamere preizkusite zakonitosti, ki jih opisuje enačba kamere z luknjico v praksi.
Kamero postavite na statično mesto s pogledom na mizo. Pred kamero na izmerjeno razdaljo od nje postavite objekt.
S programom za zajem slik iz kamere pridobite več (vsaj šest) zaporednih slik objekta pri
 čemer objekt premikajte na različne razdalje in zabeležite oddaljenost od kamere.
Nato posamezne slike odprite s python skripto in zabeležite višino objekta v številu slikovnih elementov (pomagajte si s knjižnico PyPlot (funkcija plt.ginput()), lahko pa to naredite tudi v programu za urejanje slik).
Na podlagi višine v slikovnih elementih in oddaljenosti od kamere lahko določite kakšna bo velikost v slikovnih elementih pri drugi razdalji od kamere. Preverite oceno še z dejansko meritvijo in ocenite napake.
(primer: Lonček cca 500px pri 10cm)
"""
import cv2
import matplotlib

matplotlib.use('TkAgg')  # stabilen interaktivni backend za ginput
import matplotlib.pyplot as plt
import numpy as np

# nastavitve
num_images = 6
distances_cm = [20, 30, 40, 50, 60, 70]  # oddaljenosti objekta od kamere
predict_distance = 10  # razdalja, za katero napovedujemo višino objekta

# zajem slik
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise Exception("Kamera ni dostopna")

print("2e) Zajem slik. Premikajte objekt na označene razdalje.")

for i in range(num_images):
    input(f"Pritisnite Enter, ko je objekt na razdalji {distances_cm[i]} cm: ")
    ret, frame = cap.read()
    if ret:
        filename = f"object_{i + 1}.png"
        cv2.imwrite(filename, frame)
        print(f"Slika {filename} je shranjena")
    else:
        print("Napaka pri zajemu slike")

cap.release()
cv2.destroyAllWindows()

# merjenje višine objekta v slikovnih elementih
heights_px = []

for i in range(num_images):
    img = cv2.imread(f"object_{i + 1}.png")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img_rgb)
    plt.title(f"2e) Kliknite vrh in dno objekta za sliko {i + 1}")
    points = plt.ginput(2)  # izberemo vrh in dno objekta
    plt.close()

    height = abs(points[1][1] - points[0][1])
    heights_px.append(height)

print("Višina objekta v slikovnih elementih:", heights_px)


# napoved višine pri novi razdalji
def predict_height(h1, z1, z2):
    return h1 * z1 / z2


predicted_heights = [predict_height(h, z, predict_distance) for h, z in zip(heights_px, distances_cm)]
predicted_mean = np.mean(predicted_heights)
print(f"Predvidena višina objekta pri {predict_distance} cm: {predicted_mean:.2f} px")

# preverjanje napake
measured_height = float(input(f"Vpišite dejansko višino objekta pri {predict_distance} cm (v px): "))

error = abs(predicted_mean - measured_height)
print(f"Napaka predikcije: {error:.2f} px")
