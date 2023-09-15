#!/usr/bin/env python
# coding: utf-8

# In[9]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# Wczytanie danych
data = pd.read_csv('C:/Users/konta/Desktop/uczenie maszynowe/Mnist/nrippner-mnist-handwritten-digits/MNIST_data.csv')
target = pd.read_csv('C:/Users/konta/Desktop/uczenie maszynowe/Mnist/nrippner-mnist-handwritten-digits/MNIST_target.csv')

# Wyświetlenie nazw kolumn w ramce danych target
print(target.columns)

# Informacje o klasach
classes = target['column_0'].unique()
num_samples_per_class = target['column_0'].value_counts()

print("Klasy: ", classes)
print("Liczba próbek w zbiorze: ", len(data))
print("Liczba próbek w każdej klasie: ")
print(num_samples_per_class)

# Podział na zbiór treningowy i testowy
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)

# Tworzenie i trenowanie modelu regresji logistycznej
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train['column_0'])

# Przewidywanie wyników dla danych testowych
y_pred = model.predict(X_test)

# Obliczenie dokładności modelu
accuracy = accuracy_score(y_test, y_pred)
print('Dokładność modelu: ', accuracy)

# Obliczenie macierzy pomyłek
confusion_matrix = confusion_matrix(y_test, y_pred)
print('Macierz pomyłek:')
print(confusion_matrix)



# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(9,9))
sns.heatmap(confusion_matrix,annot=True, fmt='.1f', linewidths=.5, cmap='Blues_r')
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
all_sample_table = 'Accuracy Score: {0}'.format(accuracy)
plt.title(all_sample_table,size=15)


# In[ ]:




