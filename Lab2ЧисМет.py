import csv
import matplotlib.pyplot as plt
import numpy as np



#  створення файлу з даними
def prepare_data():
    data = [
        ['n', 't'],
        [1000, 3],
        [2000, 5],
        [4000, 11],
        [8000, 28],
        [16000, 85]
    ]
    with open('data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)

#  РЕАЛІЗАЦІЯ ФУНКЦІЙ

# Зчитування вхідних даних з текстового файлу
def read_data(filename):
    x, y = [], []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            x.append(float(row['n']))
            y.append(float(row['t']))
    return x, y

# --- ДОДАТИ ПЕРЕД prepare_data() ---
def print_divided_diff_table(x, y):
    n = len(x)
    table = np.zeros((n, n + 1))
    for i in range(n):
        table[i][0], table[i][1] = x[i], y[i]

    # Розрахунок таблиці за ітеративною формулою
    for j in range(2, n + 1):
        for i in range(n - j + 1):
            table[i][j] = (table[i+1][j-1] - table[i][j-1]) / (x[i+j-1] - x[i])

    headers = ["x", "f(x)"] + [f"{k}-DD" for k in range(1, n)]
    print(" ТАБЛИЦЯ РОЗДІЛЕНИХ РІЗНИЦЬ (Варіант 1) ".center(85))
    print(" | ".join(f"{h:<12}" for h in headers))
    print("-" * 85)
    for i in range(n):
        row = [f"{table[i][j]:<12.6f}" if j < n - i + 1 else " " * 12 for j in range(n + 1)]
        print(" | ".join(row))
    print("="*85 + "\n")

# Знаходження значення допоміжної функції omega_k(x)
# (Реалізація факторіальних многочленів)
def get_omega(x_nodes, k, x_val):
    res = 1.0
    for i in range(k):
        res *= (x_val - x_nodes[i]) # ТУТ реалізовано формулу похибки
    return res

# Знаходження розділеної різниці довільного порядку
def divided_diff(x, y):
    k = len(x)
    res = 0
    for i in range(k):
        denominator = 1.0
        for j in range(k):
            if i != j:
                denominator *= (x[i] - x[j])
        if denominator == 0: continue
        res += y[i] / denominator
    return res

# Знаходження значення інтерполяційного многочлена Ньютона
# Реалізація методу Ньютона
def newton_poly(x_nodes, y_nodes, x_val, n_order):
    x_active = x_nodes[:n_order]
    y_active = y_nodes[:n_order]

    res = y_active[0]  # f0
    for k in range(1, n_order):
        #  розділена різниця * omega_k-1(x)
        df = divided_diff(x_active[:k + 1], y_active[:k + 1])
        w = get_omega(x_active, k, x_val)
        res += df * w
    return res

newton_predict = newton_poly

prepare_data()
x_data, y_data = read_data("data.csv")
n_total = len(x_data)
target_n = 6000

# >>> ДОДАЙ ЦЕЙ РЯДОК СЮДИ <<<
print_divided_diff_table(x_data, y_data)

# Табуляція та запис у текстовий файл
# Крок h = (b-a)/(20*n)
a, b = min(x_data), max(x_data)
h_tab = (b - a) / (20 * n_total)
x_tab = np.arange(a, b + h_tab, h_tab)



with open("tabulation_results.txt", "w", encoding="utf-8") as f:
    f.write(f"{'x (n)':<10} | {'Nn(x)':<12} | {'wn(x)':<15}\n")
    f.write("-" * 45 + "\n")
    for xi in x_tab:
        nn = newton_poly(x_data, y_data, xi, n_total)
        wn = get_omega(x_data, n_total, xi)
        f.write(f"{xi:<10.1f} | {nn:<12.4f} | {wn:<15.2e}\n")


# Прогноз для n=6000
prediction = newton_predict(x_data, y_data, target_n, n_total)
print(f"   Прогноз часу для n={target_n}: {prediction:.2f} мс")

#  Повторні обчислення
print("\n ДОСЛІДЖЕННЯ ТОЧНОСТІ (різна кількість вузлів):")
test_nodes = [3, 4, 5, 10, 20]
for count in test_nodes:
    # Беремо доступну кількість вузлів (макс. 5 для Варіанту 1)
    active = min(count, n_total)
    res = newton_poly(x_data, y_data, target_n, active)
    print(f"   Вузлів: {count:<2} | Прогноз: {res:.2f} мс")

#  Побудова графіків
plt.figure(figsize=(10, 6))
plt.scatter(x_data, y_data, color='red', label='Експериментальні точки')
x_fine = np.linspace(a, b, 200)
y_fine = [newton_poly(x_data, y_data, val, n_total) for val in x_fine]
plt.plot(x_fine, y_fine, 'b-', label='Крива Ньютона')
plt.axvline(x=target_n, color='green', linestyle='--', label=f'Прогноз n={target_n}')

plt.title("Прогнозування продуктивності алгоритму (Варіант 1)")
plt.xlabel("Розмір вхідних даних (n)")
plt.ylabel("Час виконання (t, мс)")
plt.legend()
plt.grid(True)



# Побудова графіків згідно


# Створюємо два графіки один під одним
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
plt.subplots_adjust(hspace=0.4)

# Експериментальні точки та крива Ньютона
ax1.scatter(x_data, y_data, color='red', label='Експериментальні точки f(n)')
x_fine = np.linspace(a, b, 300)
y_fine = [newton_poly(x_data, y_data, val, n_total) for val in x_fine]
ax1.plot(x_fine, y_fine, 'b-', label='Інтерполяційна крива Nn(x)')
ax1.axvline(x=target_n, color='green', linestyle='--', label=f'Прогноз n={target_n}')

ax1.set_title("Інтерполяція продуктивності алгоритму  ")
ax1.set_xlabel("Розмір вхідних даних (n)")
ax1.set_ylabel("Час виконання (t, мс)")
ax1.legend()
ax1.grid(True)

#  Функція похибки wn(x)
# Ця функція показує розподіл теоретичної похибки між вузлами
y_omega = [get_omega(x_data, n_total, val) for val in x_fine]
ax2.plot(x_fine, y_omega, 'purple', label='Функція похибки wn(x)')
ax2.axhline(0, color='black', linewidth=1) # Лінія нуля

ax2.set_title("Графік функції похибки wn(x) ")
ax2.set_xlabel("n")
ax2.set_ylabel("Значення wn(x)")
ax2.legend()
ax2.grid(True)





plt.show()

