###################################################################################
###                              LIBRERÍAS                                      ###
###################################################################################

import numpy as np
import cv2
import matplotlib.pyplot as plt
from math import exp, pi
from scipy import signal

###################################################################################
###                           CARGANDO IMAGENES                                 ###
###################################################################################

def cargar_imagenes():
    imagenes = {}
    for i in range(6, 56):
        if i <10:
            p = f"0{i}"
        else:
            p = i
        imagen = cv2.imread(f"images/ISIC_00243{p}.jpg")
        segmentation = cv2.imread(f"images/ISIC_00243{p}_segmentation.png")
        imagenes[f"imagen {i}"] = [imagen, segmentation]
    return imagenes

''' 
def cargar_test(l):
    l:lista
'''
###################################################################################
###                          CALCULO DE ERROR                                   ###
###################################################################################
def calculando_error(ideal, real):
    N, M = ideal.shape
    TP, TN, FP, FN = 0, 0, 0, 0 
    for i in range(N):
        for j in range(M):
            if ideal[i,j] == real[i,j] == 1:
                TP += 1
            elif ideal[i,j] == real[i,j] == 0:
                TN += 1
            elif ideal[i,j] != real[i,j] == 1:
                FP += 1
            elif ideal[i, j] != real[i,j] == 0:
                FN += 1
    TPR = TP / (TP+FN)
    FPR = FP/(FP+TN)
    return TPR, FPR
              
                
  
######### Función Segmentar ###########
def segmentar(I, l):
    ## Permite obtener la segmentación de una imagen en escala de grises para un rango de valores entre 0 y 255
    ## I: imagen en escala de grises; l = intervalo de segmentación de tipo [v_min, v_max]
    N, M = I.shape
    J = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if I[i,j] >= l[0] and I[i,j]<=l[-1]:
                J[i,j] = 1
    return J 
      
######### Escala de Grises ###########
def escalas_de_grises(dic_imagenes):
    escala_grises = []
    for llave in dic_imagenes:
        gray = cv2.cvtColor(dic_imagenes[llave][0], cv2.COLOR_BGR2GRAY)
        escala_grises.append(gray)
    return escala_grises

######### Filtro Mediana ###########
def filtro_mediana(imagenes):
    median_image = []
    for imagen in imagenes:
        n = int(imagen.shape[0] * imagen.shape[1] / 10000)
        median_image.append(cv2.medianBlur(imagen, n))
    return median_image

   
def dog_filter(image, s = 1):
    filtro = []
    N, M = image.shape
    deg_image = np.zeros((N, N))
    for i in range(-3*s, 3*s +1):
        dog = round(- (i * exp(- i ** 2 / (2 * s ** 2))) / ((2 * pi) ** (1/2) * s ** 3), 2)
        filtro.append(dog)
    mask_x = np.array(filtro).reshape(1, 6*s + 1)
    mask_y = np.transpose(mask_x)
    image_x = signal.convolve2d(image, mask_x, mode = "same")
    image_y = signal.convolve2d(image, mask_y, mode = "same")
    deg_image = image_x + image_y


    return deg_image

imagenes = cargar_imagenes()
gray_images = escalas_de_grises(imagenes)


plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(gray_images[37], cmap = "gray")
plt.title("Imagen Original")
plt.subplot(1, 2, 2)
plt.imshow(dog_filter(gray_images[37], s = 1), cmap = "gray")
plt.title("1")
plt.show()
######### Pasar Máscara ########### 

def pass_mask(image, mask):
    N, M = image.shape
    new_image = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            if mask[i, j] == 255:
                new_image[i, j] = 255
            else:
                new_image[i, j] = image[i, j]
    return new_image
    
######### Re - Pintar la imagen ############

def paint(image):
    N, M = image.shape
    hay_255 = True
    while hay_255:
        contador_nowhite = 0
        for i in range(N):
            for j in range(M):
                if image[i, j] == 255 and i != 0 and i != N - 1\
                and j != 0 and j != M - 1:
                    lista_veci = []
                    suma = 0
                    for k in range(i - 1, i + 2):
                        for z in range(j - 1, j + 2):
                            if image[k, z] != 255:
                                lista_veci.append(image[k, z])
                    for elementos in lista_veci:
                        suma += elementos
                    image[i, j] = suma / len(lista_veci)
                else:
                    contador_nowhite += 1
        if contador_nowhite == N*M:
            hay_255 = False
    return image
