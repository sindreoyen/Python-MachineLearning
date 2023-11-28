# ==========================================================
# Aprendizaje automático 
# Máster en Ingeniería Informática - Universidad de Sevilla
# Curso 2023-24
# Primer trabajo práctico
# ===========================================================

# --------------------------------------------------------------------------
# APELLIDOS: Øyen
# NOMBRE: Sindre Langås
# ----------------------------------------------------------------------------


# *****************************************************************************
# HONESTIDAD ACADÉMICA Y COPIAS: un trabajo práctico es un examen, por lo que
# debe realizarse de manera individual. La discusión y el intercambio de
# información de carácter general con los compañeros se permite (e incluso se
# recomienda), pero NO AL NIVEL DE CÓDIGO. Igualmente el remitir código de
# terceros, OBTENIDO A TRAVÉS DE LA RED o cualquier otro medio, se considerará
# plagio. 

# Cualquier plagio o compartición de código que se detecte significará
# automáticamente la calificación de CERO EN LA ASIGNATURA para TODOS los
# estudiantes involucrados. Por tanto, NO se les conservará, para
# futuras convocatorias, ninguna nota que hubiesen obtenido hasta el
# momento. SIN PERJUICIO DE OTRAS MEDIDAS DE CARÁCTER DISCIPLINARIO QUE SE
# PUDIERAN TOMAR.  
# *****************************************************************************


# IMPORTANTE: NO CAMBIAR EL NOMBRE NI A ESTE ARCHIVO NI A LAS CLASES Y MÉTODOS
# QUE SE PIDEN





# ========================
# IMPORTANTE: USO DE NUMPY
# ========================

# SE PIDE USAR NUMPY EN LA MEDIDA DE LO POSIBLE. 

import numpy as np

# SE PENALIZARÁ el uso de bucles convencionales si la misma tarea se puede
# hacer más eficiente con operaciones entre arrays que proporciona numpy. 

# PARTICULARMENTE IMPORTANTE es el uso del método numpy.dot. 
# Con numpy.dot podemos hacer productos escalares de pesos por características,
# y extender esta operación de manera compacta a dos dimensiones, cuando tenemos 
# varias filas (ejemplos) e incluso varios varios vectores de pesos.  

# En lo que sigue, los términos "array" o "vector" se refieren a "arrays de numpy".  

# NOTA: En este trabajo NO se permite usar scikit-learn (salvo en el código que
# se proporciona para cargar los datos).

# -----------------------------------------------------------------------------

# *****************************************
# CONJUNTOS DE DATOS A USAR EN ESTE TRABAJO
# *****************************************

# Para aplicar las implementaciones que se piden en este trabajo, vamos a usar
# los siguientes conjuntos de datos. Para cargar todos los conjuntos de datos,
# basta con descomprimir el archivo datos-trabajo-aa.zip y ejecutar el
# archivo carga_datos.py (algunos de estos conjuntos de datos se cargan usando
# utilidades de Scikit Learn). Todos los datos se cargan en arrays de numpy.

# * Datos sobre concesión de prestamos en una entidad bancaria. En el propio
#   archivo datos/credito.py se describe con más detalle. Se carga en las
#   variables X_credito, y_credito.   

# * Conjunto de datos de la planta del iris. Se carga en las variables X_iris,
#   y_iris.  

# * Datos sobre votos de cada uno de los 435 congresitas de Estados Unidos en
#   17 votaciones realizadas durante 1984. Se trata de clasificar el partido al
#   que pertenece un congresista (republicano o demócrata) en función de lo
#   votado durante ese año. Se carga en las variables X_votos, y_votos. 

# * Datos de la Universidad de Wisconsin sobre posible imágenes de cáncer de
#   mama, en función de una serie de características calculadas a partir de la
#   imagen del tumor. Se carga en las variables X_cancer, y_cancer.
  
# * Críticas de cine en IMDB, clasificadas como positivas o negativas. El
#   conjunto de datos que usaremos es sólo una parte de los textos. Los textos
#   se han vectorizado usando CountVectorizer de Scikit Learn. Como vocabulario, 
#   se han usado las 609 palabras que ocurren más frecuentemente en las distintas 
#   críticas. Los datos se cargan finalmente en las variables X_train_imdb, 
#   X_test_imdb, y_train_imdb,y_test_imdb.    

# * Un conjunto de imágenes (en formato texto), con una gran cantidad de
#   dígitos (de 0 a 9) escritos a mano por diferentes personas, tomado de la
#   base de datos MNIST. En digitdata.zip están todos los datos en formato
#   comprimido. Para preparar estos datos habrá que escribir funciones que los
#   extraigan de los ficheros de texto (más adelante se dan más detalles). 




# ===========================================================
# EJERCICIO 1: SEPARACIÓN EN ENTRENAMIENTO Y PRUEBA (HOLDOUT)
# ===========================================================

# Definir una función 

#           particion_entr_prueba(X,y,test=0.20)

# que recibiendo un conjunto de datos X, y sus correspondientes valores de
# clasificación y, divide ambos en datos de entrenamiento y prueba, en la
# proporción marcada por el argumento test, y conservando la correspondencia 
# original entre los ejemplos y sus valores de clasificación.
# La división ha de ser ALEATORIA y ESTRATIFICADA respecto del valor de clasificación.

# ------------------------------------------------------------------------------
# Ejemplos:
# =========

# En votos:

# In[1]: Xe_votos,Xp_votos,ye_votos,yp_votos          
#            =particion_entr_prueba(X_votos,y_votos,test=1/3)

# Como se observa, se han separado 2/3 para entrenamiento y 1/3 para prueba:
# In[2]: y_votos.shape[0],ye_votos.shape[0],yp_votos.shape[0]
# Out[2]: (435, 290, 145)

# Las proporciones entre las clases son (aprox) las mismas en los dos conjuntos de
# datos, y la misma que en el total: 267/168=178/112=89/56

# In[3]: np.unique(y_votos,return_counts=True)
# Out[3]: (array(['democrata', 'republicano'], dtype='<U11'), array([267, 168]))
# In[4]: np.unique(ye_votos,return_counts=True)
# Out[4]: (array(['democrata', 'republicano'], dtype='<U11'), array([178, 112]))
# In[5]: np.unique(yp_votos,return_counts=True)
# Out[5]: (array(['democrata', 'republicano'], dtype='<U11'), array([89, 56]))

# La división en trozos es aleatoria y, por supuesto, en el orden en el que
# aparecen los datos en Xe_votos,ye_votos y en Xp_votos,yp_votos, se preserva
# la correspondencia original que hay en X_votos,y_votos.


# Otro ejemplo con más de dos clases:

# In[6]: Xe_credito,Xp_credito,ye_credito,yp_credito               
#              =particion_entr_prueba(X_credito,y_credito,test=0.4)

# In[7]: np.unique(y_credito,return_counts=True)
# Out[7]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([202, 228, 220]))

# In[8]: np.unique(ye_credito,return_counts=True)
# Out[8]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([121, 137, 132]))

# In[9]: np.unique(yp_credito,return_counts=True)
# Out[9]: (array(['conceder', 'estudiar', 'no conceder'], dtype='<U11'),
#          array([81, 91, 88]))
# ------------------------------------------------------------------





# ===========================================
# EJERCICIO 2: REGRESIÓN LOGÍSTICA MINI-BATCH
# ===========================================


# Se pide implementar el clasificador de regresión logística mini-batch 
# a través de una clase python, que ha de tener la siguiente estructura:

# class RegresionLogisticaMiniBatch():

#    def __init__(self,normalizacion=False,
#                 rate=0.1,rate_decay=False,batch_tam=64,
#                 pesos_iniciales=None):

#          .....
         
#    def entrena(self,entr,clas_entr,n_epochs=1000,
#                reiniciar_pesos=False):

#         ......

#     def clasifica_prob(self,E):


#         ......

#     def clasifica(self,E):


#         ......
        

# Explicamos a continuación cada uno de los métodos:


# * Constructor de la clase:
# --------------------------

#  El constructor debe tener los siguientes argumentos de entrada:


#  - El parámetro normalizacion, que puede ser True o False (False por
#    defecto). Indica si los datos se tienen que normalizar, tanto para el
#    entrenamiento como para la clasificación de nuevas instancias.  La
#    normalización es una técnica que suele ser útil cuando los distintos
#    atributos reflejan cantidades numéricas de muy distinta magnitud.
#    En ese caso, antes de entrenar se calcula la media m_i y la desviación
#    típica d_i en CADA COLUMNA i (es decir, en cada atributo) de los
#    datos del conjunto de entrenamiento.  A continuación, y antes del
#    entrenamiento, esos datos se transforman de manera que cada componente
#    x_i se cambia por (x_i - m_i)/d_i. Esta MISMA transformación se realiza
#    sobre las nuevas instancias que se quieran clasificar.

#  - rate: si rate_decay es False, rate es la tasa de aprendizaje fija usada
#    durante todo el aprendizaje. Si rate_decay es True, rate es la
#    tasa de aprendizaje inicial. Su valor por defecto es 0.1.

#  - rate_decay, indica si la tasa de aprendizaje debe disminuir en
#    cada epoch. En concreto, si rate_decay es True, la tasa de
#    aprendizaje que se usa en el n-ésimo epoch se debe de calcular
#    con la siguiente fórmula: 
#       rate_n= (rate_0)*(1/(1+n)) 
#    donde n es el número de epoch, y rate_0 es la cantidad
#    introducida en el parámetro rate anterior.   

#  - batch_tam: indica el tamaño de los mini batches (por defecto 64)
#    que se usan para calcular cada actualización de pesos.
    
#  - pesos_iniciales: Si es None, los pesos iniciales se inician 
#    aleatoriamente. Si no, debe proporcionar un array de pesos que se 
#    tomarán como pesos iniciales.     

# 

# * Método entrena:
# -----------------

#  Este método es el que realiza el entrenamiento del clasificador. 
#  Debe calcular un vector de pesos, mediante el correspondiente
#  algoritmo de entrenamiento basado en ascenso por el gradiente mini-batch, 
#  para maximizar la log verosimilitud. Describimos a continuación los parámetros de
#  entrada:  

#  - entr y clas_entr, son los datos del conjunto de entrenamiento y su
#    clasificación, respectivamente. El primero es un array (bidimensional)  
#    con los ejemplos, y el segundo un array (unidimensional) con las clasificaciones 
#    de esos ejemplos, en el mismo orden. 

#  - n_epochs: número de pasadas que se realizan sobre todo el conjunto de
#    entrenamiento.

#  - reiniciar_pesos: si es True, se reinicia al comienzo del 
#    entrenamiento el vector de pesos de manera aleatoria 
#    (típicamente, valores aleatorios entre -1 y 1).
#    Si es False, solo se inician los pesos la primera vez que se
#    llama a entrena. En posteriores veces, se parte del vector de
#    pesos calculado en el entrenamiento anterior. Esto puede ser útil
#    para continuar el aprendizaje a partir de un aprendizaje
#    anterior, si por ejemplo se dispone de nuevos datos.     

#  NOTA: El entrenamiento en mini-batch supone que en cada epoch se
#  recorren todos los ejemplos del conjunto de entrenamiento,
#  agrupados en grupos del tamaño indicado. Por cada uno de estos
#  grupos de ejemplos se produce una actualización de los pesos. 
#  Se pide una VERSIÓN ESTOCÁSTICA, en la que en cada epoch se asegura que 
#  se recorren todos los ejemplos del conjunto de entrenamiento, 
#  en un orden ALEATORIO, aunque agrupados en grupos del tamaño indicado. 


# * Método clasifica_prob:
# ------------------------

#  Método que devuelve el array de correspondientes probabilidades de pertenecer 
#  a la clase positiva (la que se ha tomado como clase 1), para cada ejemplo de un 
#  array E de nuevos ejemplos.


        
# * Método clasifica:
# -------------------
    
#  Método que devuelve un array con las correspondientes clases que se predicen
#  para cada ejemplo de un array E de nuevos ejemplos. La clase debe ser una de las 
#  clases originales del problema (por ejemplo, "republicano" o "democrata" en el 
#  problema de los votos).  


# Si el clasificador aún no ha sido entrenado, tanto "clasifica" como
# "clasifica_prob" deben devolver una excepción del siguiente tipo:

class ClasificadorNoEntrenado(Exception): pass




# Ejemplos de uso:
# ----------------



# CON LOS DATOS VOTOS:
        
#   

# En primer lugar, separamos los datos en entrenamiento y prueba (los resultados pueden
# cambiar, ya que esta partición es aleatoria)

        
# In [1]: Xe_votos,Xp_votos,ye_votos,yp_votos            
#            =particion_entr_prueba(X_votos,y_votos)

# Creamos el clasificador:
        
# In [2]: RLMB_votos=RegresionLogisticaMiniBatch()

# Lo entrenamos sobre los datos de entrenamiento:

# In [3]: RLMB_votos.entrena(Xe_votos,ye_votos)

# Con el clasificador aprendido, realizamos la predicción de las clases
# de los datos que estan en test:

# In [4]: RLMB_votos.clasifica_prob(Xp_votos)
# array([3.90234132e-04, 1.48717603e-11, 3.90234132e-04, 9.99994374e-01, 9.99347533e-01,...]) 
        
# In [5]: RLMB_votos.clasifica(Xp_votos)
# Out[5]: array(['democrata', 'democrata', 'democrata','republicano',... ], dtype='<U11')

# Calculamos la proporción de aciertos en la predicción, usando la siguiente 
# función que llamaremos "rendimiento".

def rendimiento(clasif,X,y):
    return sum(clasif.clasifica(X)==y)/y.shape[0]
        
# In [6]: rendimiento(RLMB_votos,Xp_votos,yp_votos)
# Out[6]: 0.9080459770114943    

# ---------------------------------------------------------------------

# CON LOS DATOS DEL CÀNCER
        
# Hacemos un experimento similar al anterior, pero ahora con los datos del 
# cáncer de mama, y usando normalización y disminución de la tasa         

# In[7]: Xe_cancer,Xp_cancer,ye_cancer,yp_cancer           
#           =particion_entr_prueba(X_cancer,y_cancer)


# In[8]: RLMB_cancer=RegresionLogisticaMiniBatch(normalizacion=True,rate_decay=True)

# In[9]: RLMB_cancer.entrena(Xe_cancer,ye_cancer)

# In[9]: RLMB_cancer.clasifica_prob(Xp_cancer)
# Out[9]: array([9.85046885e-01, 8.77579844e-01, 7.81826115e-07,..])

# In[10]: RLMB_cancer.clasifica(Xp_cancer)
# Out[10]: array([1, 1, 0,...])

# In[11]: rendimiento(RLMB_cancer,Xp_cancer,yp_cancer)
# Out[11]: 0.9557522123893806


# =================================================
# EJERCICIO 3: IMPLEMENTACIÓN DE VALIDACIÓN CRUZADA
# =================================================

# Este ejercicio vale 2 PUNTOS (SOBRE 10) pero se puede saltar, sin afectar 
# al resto del trabajo. Puede servir para el ajuste de parámetros en los ejercicios 
# posteriores, pero si no se realiza, se podrían ajustar siguiendo el método "holdout" 
# implementado en el ejercicio 1. 

# La técnica de validación cruzada que se pide en este ejercicio se explica
# en el tema "Evaluación de modelos".     

# Definir una función: 

#  rendimiento_validacion_cruzada(clase_clasificador,params,X,y,n=5)

# que devuelve el rendimiento medio de un clasificador, mediante la técnica de
# validación cruzada con n particiones. Los arrays X e y son los datos y la
# clasificación esperada, respectivamente. El argumento clase_clasificador es
# el nombre de la clase que implementa el clasificador. El argumento params es
# un diccionario cuyas claves son nombres de parámetros del constructor del
# clasificador y los valores asociados a esas claves son los valores de esos
# parámetros para llamar al constructor.

# INDICACIÓN: para usar params al llamar al constructor del clasificador, usar
# clase_clasificador(**params)  

# ------------------------------------------------------------------------------
# Ejemplo:
# --------
# Lo que sigue es un ejemplo de cómo podríamos usar esta función para
# ajustar el valor de algún parámetro. En este caso aplicamos validación
# cruzada, con n=5, en el conjunto de datos del cáncer, para estimar cómo de
# bueno es el valor batch_tam=16 con rate_decay en regresión logística mini_batch.
# Usando la función que se pide sería (nótese que debido a la aleatoriedad, 
# no tiene por qué coincidir exactamente el resultado):

# >>> rendimiento_validacion_cruzada(RegresionLogisticaMiniBatch,         
#             {"batch_tam":16,"rate_decay":True},Xe_cancer,ye_cancer,n=5)
# 0.9121095227289917


# El resultado es la media de rendimientos obtenidos entrenando cada vez con
# todas las particiones menos una, y probando el rendimiento con la parte que
# se ha dejado fuera. Las particiones deben ser aleatorias y estratificadas. 
 
# Si decidimos que es es un buen rendimiento (comparando con lo obtenido para
# otros valores de esos parámetros), finalmente entrenaríamos con el conjunto de
# entrenamiento completo:

# >>> LR16=RegresionLogisticaMiniBatch(batch_tam=16,rate_decay=True)
# >>> LR16.entrena(Xe_cancer,ye_cancer)

# Y daríamos como estimación final el rendimiento en el conjunto de prueba, que
# hasta ahora no hemos usado:
# >>> rendimiento(LR16,Xp_cancer,yp_cancer)
# 0.9203539823008849

#------------------------------------------------------------------------------





# ===================================================
# EJERCICIO 4: APLICANDO LOS CLASIFICADORES BINARIOS
# ===================================================



# Usando los dos modelos implementados en el ejercicio 3, obtener clasificadores 
# con el mejor rendimiento posible para los siguientes conjunto de datos:

# - Votos de congresistas US
# - Cáncer de mama 
# - Críticas de películas en IMDB

# Ajustar los parámetros para mejorar el rendimiento. Si se ha hecho el ejercicio 3, 
# usar validación cruzada para el ajuste (si no, usar el "holdout" del ejercicio 1). 

# Mostrar el proceso realizado en cada caso, y los rendimientos finales obtenidos. 




# =====================================
# EJERCICIO 5: CLASIFICACIÓN MULTICLASE
# =====================================

# Técnica "One vs Rest" (Uno frente al Resto)
# -------------------------------------------


# Se pide implementar la técnica "One vs Rest" (Uno frente al Resto),
# para obtener un clasificador multiclase a partir del clasificador
# binario definido en el apartado anterior.


#  En concreto, se pide implementar una clase python
#  RegresionLogisticaOvR con la siguiente estructura, y que implemente
#  el entrenamiento y la clasificación siguiendo el método "One vs
#  Rest" tal y como se ha explicado en las diapositivas del módulo.

 

# class RegresionLogisticaOvR():

#    def __init__(self,normalizacion=False,rate=0.1,rate_decay=False,
#                 batch_tam=64):

#          .....
         
#    def entrena(self,entr,clas_entr,n_epochs=1000):

#         ......

#    def clasifica(self,E):


#         ......
        


#  Los parámetros de los métodos significan lo mismo que en el
#  apartado anterior.

#  Un ejemplo de sesión, con el problema del iris:


# --------------------------------------------------------------------

# In[1] Xe_iris,Xp_iris,ye_iris,yp_iris          
#            =particion_entr_prueba(X_iris,y_iris,test=1/3)

# >>> rl_iris=RL_OvR(rate=0.001,batch_tam=20)

# >>> rl_iris.entrena(Xe_iris,ye_iris)

# >>> rendimiento(rl_iris,Xe_iris,ye_iris)
# 0.9797979797979798

# >>> rendimiento(rl_iris,Xp_iris,yp_iris)
# >>> 0.9607843137254902
# --------------------------------------------------------------------




# ==============================================
# EJERCICIO 6: APLICACION A PROBLEMAS MULTICLASE
# ==============================================


# ---------------------------------------------------------
# 6.1) Conjunto de datos de la concesión de crédito
# ---------------------------------------------------------

# Aplicar la implementación del apartado anterior, para obtener un
# clasificador que aconseje la concesión, estudio o no concesión de un préstamo,
# basado en los datos X_credito, y_credito. Ajustar adecuadamente los parámetros. 

# NOTA IMPORTANTE: En este caso concreto, los datos han de ser transformados, 
# ya que los atributos de este conjunto de datos no son numéricos. Para ello, usar la llamada 
# "codificación one-hot", descrita en el tema "Preprocesado e ingeniería de características".
# Se pide implementar esta transformación (directamete, SIN USAR Scikt Learn ni Pandas). 


     
                


# ---------------------------------------------------------
# 6.2) Clasificación de imágenes de dígitos escritos a mano
# ---------------------------------------------------------


#  Aplicar la implementación o implementaciones del apartado anterior, para obtener un
#  clasificador que prediga el dígito que se ha escrito a mano y que se
#  dispone en forma de imagen pixelada, a partir de los datos que están en el
#  archivo digidata.zip que se suministra.  Cada imagen viene dada por 28x28
#  píxeles, y cada pixel vendrá representado por un caracter "espacio en
#  blanco" (pixel blanco) o los caracteres "+" (borde del dígito) o "#"
#  (interior del dígito). En nuestro caso trataremos ambos como un pixel negro
#  (es decir, no distinguiremos entre el borde y el interior). En cada
#  conjunto las imágenes vienen todas seguidas en un fichero de texto, y las
#  clasificaciones de cada imagen (es decir, el número que representan) vienen
#  en un fichero aparte, en el mismo orden. Será necesario, por tanto, definir
#  funciones python que lean esos ficheros y obtengan los datos en el mismo
#  formato numpy en el que los necesita el clasificador. 

#  Los datos están ya separados en entrenamiento, validación y prueba. En este
#  caso concreto, NO USAR VALIDACIÓN CRUZADA para ajustar, ya que podría
#  tardar bastante (basta con ajustar comparando el rendimiento en
#  validación). Si el tiempo de cómputo en el entrenamiento no permite
#  terminar en un tiempo razonable, usar menos ejemplos de cada conjunto.

# Ajustar los parámetros de tamaño de batch, tasa de aprendizaje y
# rate_decay para tratar de obtener un rendimiento aceptable (por encima del
# 75% de aciertos sobre test). 




