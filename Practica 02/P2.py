# -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

from matplotlib import pyplot as plt

"""
	Funciones Utiles
"""

# Leer Imagenes
def readImage(uri, typeImg = 0):
    # Tipo 0 = Escala de Grises, Tipo 1 = Escala de Colores
    if (typeImg == 0) or (typeImg == 1):
        return cv2.imread(uri, typeImg)
    else:
        print('Tipo de Imagen no Valido')
        sys.exit()

# Mostrar Imagenes
def displayImage(img, name = 'Imagen'):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Mostrar la informacion de la Matriz de la Imagen
def showImageMatrixData(uri, name = 'Matriz Imagen', typeImg = 0):
    if (typeImg == 0) or (typeImg == 1):
        img = readImage(uri, typeImg)
        print(name)
        print('\n', img, '\n')
        print('Tamaño de la Imagen: ', img.size)
        print('Tipo de Imagen: ', img.dtype)
    else:
        print('Tipo de Imagen no Valido')
        sys.exit()
    
# Mostrar Multiples Imagenes en una Ventana OpenCV
def displayMultipleImage(images, position = 1, name = 'Imagen'):
    # Posicion 0 = Concatenacion Vertical, Position 1 = Concatenacion Horizontal
    if (position == 0) or (position == 1):
        for i in range(0, len(images)):
            # Si la Imagen es en Escala de Grises, se la transforma a Matriz RGB
            if (np.size(np.shape(images[i])) == 2):
                # Es necesario que todas las Imagenes tengan el mismo numero de dimensiones
                images[i] = cv2.cvtColor(images[i], cv2.COLOR_GRAY2BGR)
            if (i == 0):
                concat_images = images[i]
            else:
                # Concatenamos las Imagenes
                concat_images = np.concatenate((concat_images, images[i]), axis = position)
        cv2.imshow(name, concat_images)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print('Posicion de la Union de Imagenes no Valida')
        sys.exit()
        
# Mostrar Multiples Imagenes en una Ventana Matplotlib
def plotMultipleImage(images, imgNames, rows, columns, name = 'Imagen'):
    fig = plt.figure(0)
    fig.canvas.set_window_title(name)
    for i in range(rows * columns):
        if (i < len(images)):
            plt.subplot(rows, columns, i+1)
            if (len(np.shape(images[i])) == 3):
                # Si la Imagen es a color
                img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                plt.imshow(img)
            else:
                # Si la Imagen es en gris
                plt.imshow(images[i], cmap = 'gray')
            plt.title(imgNames[i])
            plt.xticks([])
            plt.yticks([])
    plt.show()
     
# Modificar los Pixeles de la Imagen
def modifyPixels(uri, typeImg = 0, pixel = 50):
    # Si la imagen esta en Escala de Grises no tiene tercera dimension
    if (typeImg == 0):
        img = readImage(uri, typeImg)
        # Se obtienen los datos de anchura y altura
        h, w = np.shape(img)
        for py in range(0, h):
            for px in range(0, w):
                # Si supera el nuevo pixel a 255, se resetea
                if (pixel + img[py][px]) > 255:
                    img[py][px] = (img[py][px] + pixel) - 255 
                else:
                    img[py][px] += pixel
        displayImage(img)
    # Si la imagen esta en Escala de Colores tiene tercera dimension
    elif (typeImg == 1):
        img = readImage(uri, typeImg) 
        h, w, bpp = np.shape(img)
        for py in range(0, h):
            for px in range(0, w):
                # Se recorre la tercera dimension
                for i in range(0, bpp):
                    # Si supera el nuevo pixel a 255, se resetea
                    if (pixel + img[py][px][i]) > 255:
                        img[py][px][i] = (img[py][px][i] + pixel) - 255
                    else:
                        img[py][px][i] += pixel
        displayImage(img)
    else:
        print('Tipo de Imagen no Valido')
        sys.exit()

# Escalar Imagenes al mismo tamaño añadiendo borde
def resizeImagesPadding(images):
    maxRows = 0
    maxCols = 0
    # Se obtiene el maximo ancho y alto
    for i in range(len(images)):
        img = images[i]
        if(len(img) > maxRows):
            maxRows = len(img)
        if(len(img[0]) > maxCols):
            maxCols = len(img[0])
    # Se agrega un borde a cada imagen
    # El tamaño es igual al maximo ancho/alto menos el ancho/alto de la Imagen
    for i in range(len(images)):
        img = images[i]
        numRows = (maxRows-len(img))/2
        numCols = (maxCols-len(img[0]))/2
        if(numRows % 2 == 0):
            top = bottom = int(numRows)
        else:
            top = int(numRows+0.5)
            bottom = int(numRows-0.5)
        if(numCols % 2 == 0):
            left = right = int(numCols)
        else:
            left = int(numCols+0.5)
            right = int(numCols-0.5)
        # Se añade el borde
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT)
        # Se reemplaza la Imagen
        images[i] = img
    return images

# Encadena Multiples Imagenes en una Ventana OpenCV para Piramides
def construct_pyramid(imgA , imgB):
    # Se obtienen las filas y columnas
    heightA, weightA = imgA.shape
    heightB, weightB = imgB.shape
    # Se crea la nueva Imagen inicializada a ceros
    imagen = np.zeros((max(heightA, heightB), weightA + weightB),  np.uint8)
    # Se concatenan las Imagenes
    imagen[:heightA, :weightA] = imgA
    imagen[:heightB, weightA:weightA + weightB] = imgB
    return imagen

# Escalar Imagenes al mismo tamaño
def resizeImages(images):
    minRows = 999999999999
    minCols = 999999999999
    # Se obtiene el minimo ancho y alto
    for i in range(len(images)):
        img = images[i]
        if(len(img) < minRows):
            minRows = len(img)
        if(len(img[0]) < minCols):
            minCols = len(img[0])
    # Se ajustan las Imagenes a la Imagen mas pequeña
    for i in range(len(images)):
        img = images[i]
        images[i] = cv2.resize(img, (minRows, minCols))
    return images

# Recortar Imagenes al mismo tamaño
def cropImages(images):
    minRows = 999999999999
    minCols = 999999999999
    # Se obtiene el minimo ancho y alto
    for i in range(len(images)):
        img = images[i]
        if(len(img) < minRows):
            minRows = len(img)
        if(len(img[0]) < minCols):
            minCols = len(img[0])
    # El tamaño es igual al ancho/alto de la Imagen menos el minimo ancho/alto
    for i in range(len(images)):
        img = images[i]
        numRows = (len(img)-minRows)/2
        numCols = (len(img[0])-minCols)/2
        if(numRows % 2 == 0):
            top = int(numRows)
            bottom = int(len(img))-int(numRows)
        else:
            top = int(numRows+(0.5))
            bottom = int(len(img))-int(numRows-0.5)
        if(numCols % 2 == 0):
            left = int(numCols)
            right = int(len(img[0]))-int(numCols)
        else:
            left = int(numCols+(0.5))
            right = int(len(img[0]))-int(numCols-(0.5))
        if(bottom == 0):
            top = 0
            bottom = int(len(img))
        if(right == 0):
            left = 0
            right = int(len(img[0]))
        # Se escala la Imagen
        images[i] = img[top:bottom, left:right]
    return images

"""
    Funciones Especificas de la Practica
"""

# Devuelve los keypoints de SIFT o SURF y las imagenes con los keypoints dibujados
def getSiftSurf(images, names, mode = 0, flagOption = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS):
    imagesMode = []
    keyPoints = []
    descriptors = []
    if(mode == 0): # SIFT
        print("\nSIFT Points")
        # Clase para extraer los keypoints y descriptores de SIFT
        sift = cv2.xfeatures2d.SIFT_create(nfeatures = 1500)
        for i in range(len(images)):
            # Se extraen los keypoint
            kp = sift.detect(images[i], None)
            # Numero de puntos obtenidos
            print("\tImagen ", names[i], ": ", len(kp))
            keyPoints.append(kp)
            # Se guardar la imagen con los keypoints dibujados
            imagesMode.append(cv2.drawKeypoints(images[i], kp, None, flags = flagOption))
            # Se obtiene el descriptor
            descriptors.append(sift.compute(images[i], kp))
    else: #SURF
        print("\nSURF Points")
        # Clase para extraer los keypoints y descriptores de SURF
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold = 500)
        for i in range(len(images)):
            # Se extraen los keypoint
            kp = surf.detect(images[i], None)
            # Numero de puntos obtenidos
            print("\tImagen ", names[i], ": ", len(kp))
            keyPoints.append(kp)
            # Se guardar la imagen con los keypoints dibujados
            imagesMode.append(cv2.drawKeypoints(images[i], kp, None, flags = flagOption))
            # Se obtiene el descriptor
            descriptors.append(surf.compute(images[i], kp))
    return imagesMode, keyPoints, descriptors

# Funcion para desempaquetar las octavas
def unpackOctave(keyPoints):
    # Se crea el diccionario
    dictOctave = dict()
    for kpoint in keyPoints:
        # Se obtienen la octava, la capa y la escala
        _octave = kpoint.octave
        octave = _octave & 0xFF
        layer = (_octave >> 8) & 0xFF
        if octave >= 128:
            octave |= -128
        if octave >= 0:
            scale = float(1/(1 << octave))
        else:
            scale = float(1 << -octave)
        # Se actualiza el diccionario con los datos obtenidos
        if str(octave) in dictOctave:
            dictOctave[str(octave)] = np.append(dictOctave[str(octave)], [[int(layer), scale, kpoint.angle]], axis = 0)
        else:
            dictOctave[str(octave)] = np.array([[int(layer), scale, kpoint.angle]])
    return dictOctave

# Devuelve la informacion de (Octava, Num. Elementos) y (Capas por Octava, Puntos Detectados por Capa)
def dictionaryInfo(keyPoints, title):
    # Obtenemos el diccionario
    dictOctave = unpackOctave(keyPoints)
    print(title)
    # Extraemos la informacion
    for i in dictOctave:
        print("\tOctava: ", i, ", Num. Elementos: ", dictOctave[i].shape)
        for j in np.unique(dictOctave[i][:, 0]):
            print("\tCapa: ", j, ", Puntos Detectados: ", dictOctave[i][dictOctave[i][:, 0] == j].shape[0])

"""
	Codigo Ejercicios
"""

def ejercicio_1(path = "data/yosemite-basic/", channel = 1):
    # Leemos las imagenes y les asignamos nombres
    images = [readImage(path+"Yosemite1.jpg", channel), readImage(path+"Yosemite2.jpg", channel)]
    names = ["Yosemite1", "Yosemite2"]
    # Obtenemos los keypoints y las imagenes con los keypoints dibujados
    # Tambien obtenemos los descriptores a partir de los keypoints
    imagesSift, kpSift, descriptorsSift = getSiftSurf(images, names, 0)
    imagesSurf, kpSurf, descriptorsSurf = getSiftSurf(images, names, 1)
    # Mostramos las imagenes
    displayMultipleImage(imagesSift, 1, 'Imagenes SIFT')
    displayMultipleImage(imagesSurf, 1, 'Imagenes SURF')
    # Obtenemos informacion de las Octavas y sus elementos
    # Tambien de las capas y los puntos detectados
    for i in range(len(kpSift)):
        dictionaryInfo(kpSift[i], "\nSIFT Points "+names[i])
    for i in range(len(kpSurf)):
        dictionaryInfo(kpSurf[i], "\nSURF Points "+names[i])

def ejercicio_2():
    print()

def ejercicio_3():
    print()
    
def ejercicio_4():
    print()

"""
	Codigo Principal
"""

def main():
    pathMosaico = "data/mosaico/"
    pathTablero = "data/tablero/"
    pathYosemiteBasic = "data/yosemite-basic/"
    pathYosemiteFull = "data/yosemite-full/"
    
    ### Ejercicio 01
    ejercicio_1(pathYosemiteBasic, 1)
    input("\nPulsa Enter para continuar la ejecucion:\n")
    
    ### Ejercicio 02
    #ejercicio_2()
    #input("\nPulsa Enter para continuar la ejecucion:\n")
    
    ### Ejercicio 03
    #ejercicio_3()
    #input("\nPulsa Enter para continuar la ejecucion:\n")
    
    ### Ejercicio 04
    #ejercicio_4()
    
if __name__ == "__main__":
    main()