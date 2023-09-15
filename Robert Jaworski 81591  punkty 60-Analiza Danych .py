#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Bilbliotek
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#wczytywanie danych
dane = pd.read_csv('C:/Users/vdi-terminal/Downloads/breast-cancer-wisconsin.data')
#Wydruk danych 
print (dane)


#Funckje pomocnicze
def ClassCheck(dana ,klasa):
    if (dana == klasa):
        return True
    return False
#funkcja sprawdzania brakujących danych
def isAnyNull(tablica):
    for zmienna in tablica:
        if(zmienna=="?"):
            return True
    return False


# In[2]:


#opisy kolumn
columny=("Id","Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape","Marginal Adhesion","Single Epithelial Cell Size","Bare Nuclei","Bland Chromatin","Normal Nucleoli","Mitoses","Class")
dane.columns = columny
print(dane["Id"])
print("Klasy: benign (niezłośliwy) , malignant  (złośliwy)")


# In[3]:


#podział danych pod względem wyniku
iloscZlosliwych=0

iloscNiezlosliwych=0

for dana in dane["Class"]:
    if (ClassCheck(dana,4)):
        iloscZlosliwych+=1
    else:
        iloscNiezlosliwych+=1
print("Ilosc złośliwych =" + str(iloscZlosliwych))
print("Ilosc nie złośliwych" + str(iloscNiezlosliwych))
print("łącznie : " + str(iloscZlosliwych+iloscNiezlosliwych))


# In[4]:


#odwołanie do funkcji sprawdzania niewłasciwych danych
for label in columny:
    if (isAnyNull(dane[label])):
        print("Brakuje danych w kolumnie"+ str(label))


# In[5]:


#wybieramy 3 kolumny zgodnie z zadaniem
wybraneLabel =["Clump Thickness","Uniformity of Cell Size","Uniformity of Cell Shape"]


# In[6]:


#analiza danych

for z in wybraneLabel:
    
    print("Dane: dla" +str(z))
    #srednia
    print("Średnia: "+str(dane[z].mean()))
    #mediana
    print("Mediana: "+str(dane[z].median()))
    #maksymalna wartosc
    print("Maksymalna wartość występująca: "+str(pd.DataFrame.max(dane[z])))
    #min Wartosc
    print("minimalna wartość występująca: "+str(pd.DataFrame.min(dane[z])))
    #sttandartowe odchylenie
    print("Standartowe odchylenie " +str(pd.DataFrame.std(dane[z])))
    #Kwartyle
    print("Kwartyle "+str(dane[z].quantile([0.25,0.75],interpolation='nearest')))
    
    print("")


# In[7]:


#wykresy
print("Histogramy: ")
for z in wybraneLabel:
    plt.figure()
    plt.title(z)
    plt.hist(dane[z])


# In[8]:


print("Pudełkowy: ")
for z in wybraneLabel:
    plt.figure()
    plt.title(z)
    plt.boxplot(dane[z])


# In[9]:


print("Korelacje: ")

# plt.plot(dane[wybraneLabel[0]],"r.")
# plt.plot(dane[wybraneLabel[1]],"b.")
daneniegrozne=dane[dane['Class']==2]
danezlosliwe=dane[dane['Class']==4]

plt.title("Niegroźne")
plt.xlabel(wybraneLabel[0])
plt.ylabel(wybraneLabel[1])
plt.plot(daneniegrozne[wybraneLabel[0]],daneniegrozne[wybraneLabel[1]],"r.")



# In[12]:


#
plt.title("Złośliwe")
plt.xlabel(wybraneLabel[0])
plt.ylabel(wybraneLabel[1])
plt.plot(danezlosliwe[wybraneLabel[0]],danezlosliwe[wybraneLabel[1]],"r.")


# In[ ]:





# In[ ]:




