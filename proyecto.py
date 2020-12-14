###################################################################################
###                              LIBRERÍAS                                      ###
###################################################################################

import numpy as np
from numpy.linalg import norm
from gap_statistic import OptimalK
import cv2
import matplotlib.pyplot as plt
from skimage import filters
from skimage.filters import threshold_otsu, threshold_yen
from skimage.exposure import rescale_intensity, equalize_hist


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
        
    TP = 0
    TN = 0
    FP = 0
    FN = 0 
    for i in range(N):
        for j in range(M):
            if ideal[i,j] == 1 and real[i,j] == 1:
                TP += 1
                print('TP')
            elif ideal[i,j] == 0 and real[i,j] == 0:
                TN += 1
                print('TN')
            elif ideal[i,j] == 0 and real[i,j] == 1:
                FP += 1
                print('FP')
            elif ideal[i, j] == 1 and real[i,j] == 0:
                FN += 1
                print('FN')
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


### CANNY ###

def auto_canny(image, sigma=0.8):
    
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def canny_otsu(image):
    val = threshold_otsu(image)
    return  cv2.Canny(image, 0.2*val, 0.8*val)

'''    
test_th = cv2.morphologyEx(img_test, cv2.MORPH_BLACKHAT, cv2.getStructuringElement(cv2.MORPH_CROSS,(10,10)))
img_test = cv2.GaussianBlur(img_test,(5, 5), 0.3)
'''

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
    new_image = image
    while hay_255:
        contador_nowhite = 0
        for i in range(N):
            for j in range(M):
                if new_image[i, j] == 255 and i != 0 and i != N - 1\
                and j != 0 and j != M - 1:
                    lista_veci = []
                    suma = 0
                    for k in range(i - 1, i + 2):
                        for z in range(j - 1, j + 2):
                            if new_image[k, z] != 255:

                                lista_veci.append(new_image[k, z])
                    if len(lista_veci) != 0:
                      
                        for elementos in lista_veci:
                            suma += elementos
                        new_image[i, j] = suma / len(lista_veci)
                else:
                    contador_nowhite += 1
        if contador_nowhite == N*M:
            hay_255 = False
    return new_image

###### ALGORITMO DE YEN ######

def yen(img):
    return(rescale_intensity(img, (0, threshold_yen(img)), (0, 255)))

##### MARCO ######
def marco(img, h, k):
    return img[h-1:-h, k-1:-k]

def marco_circular(img,r):
    N,M = img.shape
    aux = img
    for i in range(N):
        for j in range(M):
            if np.sqrt((i-N/2.0)**2.0 + (j-M/2.0)**2.0) > r:
                aux[i,j] = 255
    return aux
    
    

##### SELECCION #####

def seleccion(img, h, k):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    R, G, B = cv2.split(img)
        
    gray1 = yen(marco(gray,h,k))
    R1 = yen(marco(R,h,k))
    G1 = yen(marco(G,h,k))
    B1 = yen(marco(B,h,k))
    
    imagenes = {'gray': [np.std(gray1), gray], 'R': [np.std(R1), R], 'G': [np.std(G1), G], 'B': [np.std(B1), B]}
    ord = sorted(((v,k) for k,v in imagenes.items()))
    
    return marco_circular(ord[-1][0][1],350)

'''
##### CLUSTERIZACION LUNAR #####

def cl_lunar(img):
    
    N,M = img.shape
    
    data = []
    for i in range(N):
        for j in range(M):
            if img[i,j] != 255:
                data.append([i, j])
    data = np.array(data).astype(float)
    print(data.shape)
    
    optK = OptimalK(n_jobs = 32, parallel_backend='joblib')
    n_cl = optK(data, cluster_array=np.arange(1, 15))
    n_cl=1
    
    print(n_cl)
    if n_cl > 1:
        
        kmeans = KMeans(n_clusters = n_cl, tol = 1e-2, n_jobs = -1)
        kmeans.fit(data)
        
        labels = kmeans.labels_
        centers = kmeans.cluster_centers_
        
        dist = [norm(c - np.array([N/2.0, M/2.0])) for c in centers] 
        ind = dist.index(min(dist))
        print(centers[ind])
        
        return data[labels==ind]
    
    return(data)
                
'''

##### ALGORITMO DE PROCESAMIENTO FINAL ####

def process(img):
    
    # seleccion de canal RGB mejor constrate
    sel = seleccion(img, 50, 50)
    
    # depilacion
    test_canny = canny_otsu(sel)  
    ker = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
    test2 = cv2.morphologyEx(test_canny, cv2.MORPH_CLOSE, ker)
    clean_image = pass_mask(sel, test2)
    pintada = paint(clean_image).astype(np.uint8)
    '''
    '''
    # segmentacion
    
    med = cv2.medianBlur(pintada,23)
    '''
    bl = cv2.blur(pintada,(3,3))
    '''
    eq = equalize_hist(med)

    y = yen(marco_circular(eq, 300))
    val = filters.threshold_otsu(y)
    final = segmentar(y, [0, val])
    
    # mejora de segmentación
    cr = cv2.getStructuringElement(cv2.MORPH_CROSS,(13,13))
    final2 = cv2.morphologyEx(final, cv2.MORPH_CLOSE, cr)
        
    return med
    
    
imagenes = cargar_imagenes()
'''
list_img = []
for key in imagenes.keys():
    list_img.append(imagenes[key][0])

test = list_img[44]

final = process(test)
lunar = cl_lunar(final)

plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(test, cv2.COLOR_BGR2RGB))
plt.title('Original')

plt.subplot(1, 2, 2)
plt.imshow(final, cmap='gray')
plt.plot(lunar[:,1], lunar[:,0], 'r.')
plt.show()

   

plt.figure()

for key in imagenes.keys():
    
    img = imagenes[key][0]
    ideal = imagenes[key][1]
    
    final = process(img)
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f'{key}')

    plt.subplot(1, 2, 2)
    plt.imshow(final, cmap = 'gray')
    plt.title('Malena')
    plt.pause(0.05)
    
plt.show()
'''
def ds(img, r):
    vd = []
    N,M = img.shape
    for i in range(N):
        for j in range(M):
            if np.sqrt((i-N/2.0)**2.0 + (j-M/2.0)**2.0) <= r:
                vd.append(img[i,j])
    return np.std(np.array(vd))

def desv(imagenes):

    d = {}
    for key in imagenes.keys():
        img = imagenes[key][0]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sel = seleccion(img, 50, 50)
        '''
        test_canny = canny_otsu(sel)  
        ker = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
        test2 = cv2.morphologyEx(test_canny, cv2.MORPH_CLOSE, ker)
        clean_image = pass_mask(sel, test2)
        pintada = paint(clean_image).astype(np.uint8)
        '''
        d[key] = ds(sel, 300)
        print(key, d[key])
    return d

d = desv(imagenes)
sort = sorted(((v,k) for k,v in d.items()))
print(sort)

    




'''
blur = cv2.GaussianBlur(pintada,(15, 15), 4.0)
blur = blur.astype(np.uint8)
print(blur)
kernel = np.ones((3,3), np.uint8)
'''



'''
print(calculando_error(cv2.cvtColor(imagenes['imagen 37'][1], cv2.COLOR_BGR2GRAY).astype(np.uint8)/255,final.astype(np.uint8)))

plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(imagenes['imagen 37'][1], cv2.COLOR_BGR2GRAY).astype(np.uint8)/255)
plt.title('1')

plt.subplot(1, 2, 2)
plt.imshow(final, cmap='gray')
plt.title("2")
plt.show()


plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(imagenes['imagen 37'][0])
plt.title('Imagen original')

plt.subplot(2, 3, 2)
plt.imshow(test_canny, cmap='gray')
plt.title("Detección de bordes (Canny)")

plt.subplot(2, 3, 3)
plt.imshow(clean_image, cmap = 'gray')
plt.title("Desvellamiento")

plt.subplot(2, 3, 4)
plt.imshow(pintada, cmap = 'gray')
plt.title("Interpolación (Pintada)")

plt.subplot(2, 3, 5)
plt.imshow(med, cmap = 'gray')
plt.title('Filtro mediana')

plt.subplot(2, 3, 6)
plt.imshow(final, cmap = 'gray')
plt.title("Umbralización final (Otsu)")
plt.show()
'''
