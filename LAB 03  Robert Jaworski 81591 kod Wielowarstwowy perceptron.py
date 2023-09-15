#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Importowanie potrzebnych bibliotek:
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Wczytywanie odpowiedniego zbiorów danych

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='auto')


# Normalizacja
X = X / 255.

# Wydzielenie zbioru walidacyjnego
X_temp, X_dev, y_temp, y_dev = train_test_split(X, y, stratify=y, test_size=1000, random_state=42)

# Podział danych na zbiór uczący i testowy
X_train, X_test, y_train, y_test = train_test_split(X_temp, y_temp, stratify=y_temp, test_size=0.25, random_state=42)

# Utwórz model

# 200 iteracji
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', 
                    learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=True, 
                    warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, 
                    beta_2=0.999, epsilon=1e-08, n_iter_no_change=10, max_fun=15000)


# szybsze
# mlp = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200, alpha=1e-4,
#                     solver='sgd', verbose=10, random_state=1,
#                     learning_rate_init=.1)


# mlp = MLPClassifier()


# Wytrenuj model
mlp.fit(X_train, y_train)

# Oblicz dokładność
print(f"Dokładność: {mlp.score(X_test, y_test)}")

# Oblicz i wydrukuj macierz pomyłek
cm = confusion_matrix(y_test, mlp.predict(X_test))
print("Macierz pomyłek:")
print(cm)

# Wydrukuj jak zmienia się funkcja straty w kolejnych krokach uczenia (epokach)
plt.plot(mlp.loss_curve_)
plt.title('Zmiana funkcji straty w kolejnych krokach uczenia')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.show()


# Wydrukuj pierwszych pięć wzorców, dla których rozpoznawanie jest błędne
predictions = mlp.predict(X_test)
y_test_array = np.array(y_test)
X_test_array = np.array(X_test)
incorrect_indices = np.nonzero(predictions != y_test_array)[0][:5]

print("Pierwszych pięć wzorców, dla których rozpoznawanie jest błędne:")
for idx in incorrect_indices:
    plt.imshow(X_test_array[idx].reshape((28, 28)), cmap=plt.cm.gray, interpolation='nearest')
    plt.title(f"Przewidywane: {predictions[idx]}, prawdziwe: {y_test_array[idx]}")
    plt.show()
 

# Wydrukuj ile jest wzorców w każdej klasie w zbiorze walidującym
unique, counts = np.unique(y_dev, return_counts=True)
print("Ilość wzorców w każdej klasie w zbiorze walidującym:")
print(dict(zip(unique, counts)))

# Wydrukuj krzywą uczenia dla zbioru uczącego i walidującego
mlp.fit(X_train, y_train)
loss_train = mlp.loss_curve_
mlp.fit(X_dev, y_dev)
loss_dev = mlp.loss_curve_
plt.plot(loss_train, label='train')
plt.plot(loss_dev, label='dev')
plt.title('Zależność funkcji straty od kolejnych epok dla zbioru uczącego i walidującego')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()
plt.show()





# In[ ]:





# In[ ]:




