import requests
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1-2. Отримання вхідних даних через API [cite: 92, 94]
# ---------------------------------------------------------
url = "https://api.open-elevation.com/api/v1/lookup?locations=48.164214,24.536044|48.164983,24.534836|48.165605,24.534068|48.166228,24.532915|48.166777,24.531927|48.167326,24.530884|48.167011,24.530061|48.166053,24.528039|48.166655,24.526064|48.166497,24.523574|48.166128,24.520214|48.165416,24.517170|48.164546,24.514640|48.163412,24.512980|48.162331,24.511715|48.162015,24.509462|48.162147,24.506932|48.161751,24.504244|48.161197,24.501793|48.160580,24.500537|48.160250,24.500106"

response = requests.get(url)
data = response.json()
results = data["results"]
n = len(results) # [cite: 106]

# 3. Запис у текстовий файл [cite: 104]
f = open('nodes.txt', 'w', encoding='utf-8')
f.write("№ | Latitude | Longitude | Elevation (m)\n")
for i, p in enumerate(results):
    f.write(f"{i} | {p['latitude']:.6f} | {p['longitude']:.6f} | {p['elevation']:.2f}\n")
f.close()

# ---------------------------------------------------------
# 4. Відстані (Haversine) [cite: 116, 121]
# ---------------------------------------------------------
def get_dist(lat1, lon1, lat2, lon2):
    R = 6371000 # [cite: 121]
    p1, p2 = np.radians(lat1), np.radians(lat2) # [cite: 122]
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1) # [cite: 123-126]
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2 # [cite: 127-128]
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a)) # [cite: 129]

coords = [(p["latitude"], p["longitude"]) for p in results]
elevs = [p["elevation"] for p in results]

x_dist = [0]
for i in range(1, n):
    d = get_dist(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1])
    x_dist.append(x_dist[-1] + d)

x = np.array(x_dist)
y = np.array(elevs)

# ---------------------------------------------------------
# 6-9. Побудова сплайна та метод прогонки [cite: 139-144]
# ---------------------------------------------------------
def get_spline(x_in, y_in, x_out, debug=False):
    m = len(x_in)
    h = np.diff(x_in) # [cite: 16]

    # 6. Коефіцієнти матриці системи для Ci [cite: 140]
    alfa = np.zeros(m); beta = np.ones(m); hamma = np.zeros(m); delta = np.zeros(m)
    for i in range(1, m - 1):
        alfa[i] = h[i - 1] # [cite: 43]
        beta[i] = 2 * (h[i - 1] + h[i]) # [cite: 43]
        hamma[i] = h[i] # [cite: 43]
        delta[i] = 3 * ((y_in[i + 1] - y_in[i]) / h[i] - (y_in[i] - y_in[i - 1]) / h[i - 1]) # [cite: 43]

    if debug:
        print("\n--- Коефіцієнти системи (alfa, beta, hamma, delta) ---")
        for i in range(1, m-1): print(f"i={i}: a={alfa[i]:.2f}, b={beta[i]:.2f}, g={hamma[i]:.2f}, d={delta[i]:.2f}")

    # 7. Метод прогонки (пряма) [cite: 141, 53]
    A = np.zeros(m); B = np.zeros(m)
    for i in range(1, m):
        znam = alfa[i] * A[i - 1] + beta[i] # [cite: 63-64]
        A[i] = -hamma[i] / znam # [cite: 63]
        B[i] = (delta[i] - alfa[i] * B[i - 1]) / znam # [cite: 64]

    if debug:
        print("\n--- Коефіцієнти прогонки (A, B) ---")
        for i in range(m): print(f"i={i}: A={A[i]:.4f}, B={B[i]:.4f}")

    # 8. Зворотна прогонка (Ci) [cite: 142, 66]
    c_coeffs = np.zeros(m)
    c_coeffs[m - 1] = B[m - 1] # [cite: 69]
    for i in range(m - 2, -1, -1):
        c_coeffs[i] = A[i] * c_coeffs[i + 1] + B[i] # [cite: 71]

    if debug:
        print("\n--- Знайдені коефіцієнти c[i] ---")
        print(c_coeffs)

    # 9. Коефіцієнти ai, bi, di [cite: 143-144]
    a_coeffs = y_in[:-1] # [cite: 15, 36]
    b_coeffs = (y_in[1:] - y_in[:-1]) / h - (h / 3) * (c_coeffs[1:] + 2 * c_coeffs[:-1]) # [cite: 38]
    d_coeffs = (c_coeffs[1:] - c_coeffs[:-1]) / (3 * h) # [cite: 37]

    if debug:
        print("\n--- Коефіцієнти сплайна (a, b, d) ---")
        for i in range(len(a_coeffs)): print(f"i={i}: a={a_coeffs[i]:.2f}, b={b_coeffs[i]:.4f}, d={d_coeffs[i]:.6f}")

    # Обчислення значень
    y_out = []
    for val in x_out:
        idx = 0
        for j in range(m - 1):
            if x_in[j] <= val <= x_in[j + 1]: idx = j; break
        dx = val - x_in[idx]
        y_out.append(a_coeffs[idx] + b_coeffs[idx] * dx + c_coeffs[idx] * dx ** 2 + d_coeffs[idx] * dx ** 3) # [cite: 11]
    return np.array(y_out)

# ---------------------------------------------------------
# Побудова графіків та вивід [cite: 147, 148]
# ---------------------------------------------------------
x_smooth = np.linspace(x[0], x[-1], 300)
y_21 = get_spline(x, y, x_smooth, debug=True) # Основний вивід у консоль

# Вікно з похибками (Фото 2 / Пункт 12)
plt.figure("Похибка апроксимації", figsize=(8, 5))
for cnt in [10, 15, 20]:
    idx = np.linspace(0, n - 1, cnt, dtype=int)
    y_approx = get_spline(x[idx], y[idx], x_smooth)
    err = np.abs(y_21 - y_approx) #
    plt.plot(x_smooth, err, label=f'Вузлів: {cnt}')
    print(f"=== {cnt} вузлів: Max помилка = {np.max(err):.2f}, Med = {np.mean(err):.2f}")

plt.title("Похибка апроксимації epsilon"); plt.legend(); plt.grid()

# Вікно з профілями (Фото 1 / Пункт 10) [cite: 145]
plt.figure("Вплив кількості вузлів", figsize=(8, 5))
plt.plot(x, y, 'ro', label='API вузли')
plt.plot(x_smooth, y_21, 'b-', label='21 вузол (еталон)')
for cnt in [10, 15, 20]:
    idx = np.linspace(0, n - 1, cnt, dtype=int)
    plt.plot(x_smooth, get_spline(x[idx], y[idx], x_smooth), '--', label=f'{cnt} вузлів')

plt.title("Вплив кількості вузлів на точність"); plt.legend(); plt.grid()

# ---------------------------------------------------------
# 5. Дискретний профіль [cite: 137-138]
# ---------------------------------------------------------
plt.figure(" Дискретний профіль", figsize=(10, 5))
plt.plot(x, y, 'go--', label='Дискретні точки (вузли)')
plt.title("Залежність кумулятивної відстані до висоти(Заросляк-Говерла)"); plt.xlabel("Відстань (м)"); plt.ylabel("Висота (м)"); plt.grid(); plt.legend()

# ---------------------------------------------------------
# Додаткові завдання [cite: 150]
# ---------------------------------------------------------
print("\n\n--- Характеристики маршруту ---")
print(f"Загальна довжина (м): {x_dist[-1]:.2f}") #

total_ascent = sum(max(elevs[i] - elevs[i-1], 0) for i in range(1, n)) # [cite: 155]
print(f"Сумарний набір висоти (м): {total_ascent:.2f}")

total_descent = sum(max(elevs[i-1] - elevs[i], 0) for i in range(1, n)) # [cite: 157]
print(f"Сумарний спуск (м): {total_descent:.2f}")

# Аналіз градієнта [cite: 158]
grad_full = np.gradient(y_21, x_smooth) * 100 # [cite: 165]
print(f"\n--- Аналіз градієнта ---")
print(f"Максимальний підйом (%): {np.max(grad_full):.2f}") # [cite: 167]
print(f"Максимальний спуск (%): {np.min(grad_full):.2f}") # [cite: 168]
print(f"Середній градієнт (%): {np.mean(np.abs(grad_full)):.2f}") # [cite: 169]

# Механічна енергія [cite: 170]
mass = 80; g = 9.81 # [cite: 172-173]
energy = mass * g * total_ascent # [cite: 174-177]
print(f"\n--- Енерговитрати ---")
print(f"Механічна робота (кДж): {energy/1000:.2f}") # [cite: 178]
print(f"Енергія (ккал): {energy / 4184:.2f}") # [cite: 180]

# Пункт 11. Власні спостереження (вивід тексту)
print("\n--- Пункт 11. Спостереження ---")
print("1. Кількість вузлів критично впливає на деталізацію рельєфу.")
print("2. При 10 вузлах похибка ε максимальна, сплайн надто згладжує круті підйоми.")
print("3. При 20 вузлах наближення майже ідентичне еталону (21 вузол).")

plt.show()