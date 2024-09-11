#!/usr/bin/env python
# coding: utf-8

# # ¡Hola, Denisse!  
# 
# Mi nombre es Carlos Ortiz, soy code reviewer de Practicum y voy a revisar el proyecto que acabas de desarrollar.
# 
# Cuando vea un error la primera vez, lo señalaré. Deberás encontrarlo y arreglarlo. La intención es que te prepares para un espacio real de trabajo. En un trabajo, el líder de tu equipo hará lo mismo. Si no puedes solucionar el error, te daré más información en la próxima ocasión. 
# 
# Encontrarás mis comentarios más abajo - **por favor, no los muevas, no los modifiques ni los borres**.
# 
# ¿Cómo lo voy a hacer? Voy a leer detenidamente cada una de las implementaciones que has llevado a cabo para cumplir con lo solicitado. Verás los comentarios de esta forma:
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si todo está perfecto.
# </div>
# 
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# 
# <div class="alert alert-block alert-danger">
#     
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
#     
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# 
# Puedes responderme de esta forma: 
# 
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# </div>
# 
# ¡Empecemos!

# El objetivo de este proyecto es desarrollar un modelo que pueda analizar el comportamiento de los clientes y recomendar uno de los nuevos planes de Megaline: Smart o Ultra.

# Contenido:
#     
#     Leemos los datos
#     
#     Prueba de distintos modelos
#     
#     Creación del mejor modelo

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import math 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score


# Cargamos los datos y los revisamos

# In[2]:


df = pd.read_csv('/datasets/users_behavior.csv')


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Buen trabajo con la importación de datos.
# </div>

# In[3]:


df.head()


# In[4]:


df.info()


# Dado que son datos con los que ya trabajamos anteriormente, nos saltaremos el procesamiento de datos e iremos directo a crear el modelo 

# Dividimos los datos en 3, de entrenamiento, de validación y de prueba:

# In[5]:


features = df.drop(['is_ultra'], axis=1)
target = df['is_ultra']

features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=.40, random_state=12345)


# In[6]:


features_test, features_valid, target_test, target_valid = train_test_split(features_valid, target_valid, test_size=.50, random_state=12345)


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Has hecho bien con el split de los datos.
# </div>

# Haremos distintas pruebas para ver que modelo es más exacto y con que parámetros 

# Bosque aleatorio:

# In[7]:


best_est = 0
best_depth = 0
best_score=0
best_rf= None

for est in range(1, 30):
    for depth in range(1, 11):
        model = RandomForestClassifier(random_state=12345, n_estimators=est, max_depth=depth)
        model.fit(features_train, target_train)
        score = model.score(features_valid, target_valid)
        if score > best_score:
            best_score = score
            best_est = est
            best_depth = depth
            best_rf = model 
        
print("Accuracy del mejor modelo en el conjunto de validación:", best_score, ", best_depth:", best_depth, ", best_est:", best_est)


# Arbol de decisión:

# In[8]:


best_depth = 0
best_score = 0
best_tree= None
for depth in range(1, 11):
	model = DecisionTreeClassifier(random_state=12345, max_depth=depth) 
	model.fit(features_train, target_train) 
	score = model.score(features_valid, target_valid)
	if score > best_score:
		best_depth = depth
		best_score = score
		best_tree = model
        
print("Accuracy del mejor modelo en el conjunto de validacion:", best_score, ", best_depth:", best_depth)


# Regresión logica:

# In[9]:


model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train) 
score_valid = model.score(features_valid, target_valid) 

print("Accuracy del modelo de regresión logística en el conjunto de entrenamiento:", score_valid)


# Con esto concluimos que el más exacto es el bosque aleatorio, con 14 árboles y con 10 de profundidad, que tiene 0.81 de exactitud, asi que elegiremos ese modelo

# Asi que comprobaremos la calidad del modelo de bosque aleatorio usando el conjunto de prueba

# In[10]:


best_preds = best_rf.predict(features_test)
print(accuracy_score(best_preds, target_test))


# A continuación hacemos una prueba de cordura al modelo:

# In[11]:


dummy_classifier = DummyClassifier(strategy="most_frequent")

dummy_classifier.fit(features_train, target_train)

dummy_predictions = dummy_classifier.predict(features_test)

accuracy = accuracy_score(target_test, dummy_predictions)

print("Precisión del modelo Dummy:", accuracy)


# Vemos que el modelo que creamos tiene una mayor exactitud que el Dummyclassifier, por lo que supera con éxito la prueba de cordura

# Con esto, en este proyecto creamos un modelo que usando un bosque aleatorio, analiza el comportamiento de los clientes y recomienda uno de los nuevos planes de Megaline: Smart o Ultra.
# Elegimos este modelo, comparando la exactitud de varios modelos, y resultando este el mejor, fue el escogido

# <div class="alert alert-block alert-danger">
# 
# # Comentarios generales #2
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Bien, Denisse. Nos quedan algunos elementos más de forma que de fondo por resolver. He dejado comentarios adicionales para ello. Agrega una conclusión que resuma lo realizado, lo encontrado a partir de lo realizado y responda los objetivos del proyecto.
#     
# </div>

# <div class="alert alert-block alert-success">
# 
# # Comentarios generales #3
# <b>Comentario del revisor</b> <a class="tocSkip"></a>
#     
# Todo corregido. Has aprobado un nuevo proyecto.
#     
# </div>
