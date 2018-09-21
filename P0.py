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
    # El numero de Imagenes debe coincidir con el de particiones
    if (len(images) == (rows * columns)):
        fig = plt.figure(0)
        fig.canvas.set_window_title(name)
        for i in range(rows * columns):
            if (i < len(images)):
                plt.subplot(rows, columns, i+1)
                if (len(np.shape(images[i])) == 3):
                    img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                    plt.imshow(img)
                else:
                    plt.imshow(images[i], cmap = 'gray')
                plt.title(imgNames[i])
                plt.xticks([])
                plt.yticks([])
        plt.show()
    else:
        print('El numero de particiones es distinto al de Imagenes')
        sys.exit()
     
#Modificar los Pixeles de la Imagen
def modifyPixels(uri, typeImg = 0, pixel = 50):
    #Si la imagen esta en Escala de Grises no tiene tercera dimension
    if (typeImg == 0):
        img = readImage(uri, typeImg)
        #Se obtienen los datos de anchura y altura
        h, w = np.shape(img)
        for py in range(0, h):
            for px in range(0, w):
                #Si supera el nuevo pixel a 255, se resetea
                if (pixel + img[py][px]) > 255:
                    img[py][px] = (img[py][px] + pixel) - 255 
                else:
                    img[py][px] += pixel
        displayImage(img)
    #Si la imagen esta en Escala de Colores tiene tercera dimension
    elif (typeImg == 1):
        img = readImage(uri, typeImg) 
        h, w, bpp = np.shape(img)
        for py in range(0, h):
            for px in range(0, w):
                #Se recorre la tercera dimension
                for i in range(0, bpp):
                    #Si supera el nuevo pixel a 255, se resetea
                    if (pixel + img[py][px][i]) > 255:
                        img[py][px][i] = (img[py][px][i] + pixel) - 255
                    else:
                        img[py][px][i] += pixel
        displayImage(img)
    else:
        print('Tipo de Imagen no Valido')
        sys.exit()

"""
	Codigo Principal
"""

def main():
    uri = 'lena.jpg'
    
    ### Ejercicio 01
    # Se lee la Imagen en Escala de Grises y se muestra por pantalla
    displayImage(readImage(uri, 0), 'Imagen en Escala de Grises')
    # Se lee la Imagen en Escala de Colores y se muestra por pantalla
    displayImage(readImage(uri, 1), 'Imagen en Escala de Colores')
    
    input("\nPulsa Enter para continuar la ejecucion:\n")
    
    ### Ejercicio 02
    # Informacion de la Imagen en Escala de Grises
    showImageMatrixData(uri, 'Matriz Imagen en Escala de Grises', 0)
    # Informacion de la Imagen en Escala de Colores
    showImageMatrixData(uri, 'Matriz Imagen en Escala de Colores', 1)
    
    input("\nPulsa Enter para continuar la ejecucion:\n")
    
    ### Ejercicio 03
    print('Imagenes Concatenadas')
    # Creamos una Lista de Imagenes de distintos tipos
    images = []
    images.append(readImage(uri, 0))
    images.append(readImage(uri, 1))
    images.append(readImage(uri, 0))
    # Se concatenan las Imagenes en la Lista, en una sola ventana
    displayMultipleImage(images, 1, 'Imagenes Concatenadas')
    
    input("\nPulsa Enter para continuar la ejecucion:\n")
    
    ### Ejercicio 04
    modifyPixels(uri, 0, 50)
    modifyPixels(uri, 1, 50)
    
    input("\nPulsa Enter para continuar la ejecucion:\n")
    
    ### Ejercicio 05
    print('Imagenes Concatenadas con Titulos')
    # Creamos una Lista de Imagenes de distintos tipos
    # Creamos otra lista con los nombres de las Imagenes
    images = []
    imgNames = []
    images.append(readImage(uri, 0))
    imgNames.append('Lena1')
    images.append(readImage(uri, 1))
    imgNames.append('Lena2')
    images.append(readImage(uri, 0))
    imgNames.append('Lena3')
    # Se concatenan las Imagenes en la Lista, en una sola ventana
    plotMultipleImage(images, imgNames, 1, 3, 'Imagenes Concatenadas')

if __name__ == "__main__":
	main()