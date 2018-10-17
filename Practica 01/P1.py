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
    # Se igualan los tamaños de las Imagenes
    images = resizeImages(images)
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
    # Se igualan los tamaños de las Imagenes
    images = resizeImages(images)
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
        # Se reemplaza la Imagen
        images[i] = img[top:bottom, left:right]
    return images

# Mostrar Multiples Graficas en una Ventana Matplotlib
def plotMultipleGraph(data, dataNames, rows, columns, name = 'Grafica'):
    # El numero de Graficas debe coincidir con el de particiones
    fig = plt.figure(0)
    fig.canvas.set_window_title(name)
    # Situa las Graficas secuencialmente
    for i in range(rows * columns):
        if (i < len(data)) and (i % 2 == 0):
            plt.subplot(rows, columns, i+1)
            plt.plot(data[i], 'b-', data[i], 'bo')
            plt.title(dataNames[i])
        else:
            plt.subplot(rows, columns, i+1)
            plt.pcolormesh(data[i-1].dot(data[i].T), edgecolors = 'k', linewidths = 3)
            plt.title(dataNames[i])
    plt.show()

"""
    Funciones Especificas de la Practica
"""

# Convolucion Gaussiana (Ejercicio 1A)
def gaussian_blur(img, size = (0, 0), sigma = 0, border = cv2.BORDER_DEFAULT):
	return cv2.GaussianBlur(img, size, sigma, border)
	
# Mascaras 1D (Ejercicio 1B)
def derive_convolution(derivX = 0, derivY = 0, size = 7, normal = True):
	#Ksize = 1, 3, 5, 7
    if ((size == 1) or (size == 3) or (size == 5) or (size == 7)):
        return cv2.getDerivKernels(dx = derivX, dy = derivY, ksize = size, normalize = normal)
    #Si el Ksize no es valido se obtiene error
    else:
        print('El tamaño debe ser 1, 3, 5 o 7')
        sys.exit()

# Laplaciana de Gaussiana
def laplacian_gaussian(img, sigma = 0, border = cv2.BORDER_DEFAULT, size = (0, 0), k_size = 7, depth = 0, scaler = 1, delt = 0):
    #Depth = 0, 2, 5
    img = cv2.copyMakeBorder(img, k_size, k_size, k_size, k_size, border)
    blur = gaussian_blur(img, size, sigma, border)
    return cv2.Laplacian(blur, depth, ksize = k_size, scale = scaler, delta = delt, borderType = border)

# Mascara Separable
def separable_filter(img, kernel_X = 0, kernel_Y = 0, border_T = cv2.BORDER_DEFAULT, depth = 0, delt = 0, anch = (-1, -1)):
    return cv2.sepFilter2D(img, depth, kernel_X, kernel_Y, delta = delt, borderType = border_T, anchor = anch)

# Convolucion con Derivadas
def derive(img, derivX = 0, derivY = 0, sigma = 0, border = cv2.BORDER_DEFAULT, size = 7, depth = 0, delt = 0):
    X, Y = derive_convolution(derivX, derivY, size)
    img = cv2.copyMakeBorder(img, size, size, size, size, border)
    gaussian = cv2.getGaussianKernel(size, sigma)
    blur = gaussian_blur(img, (size, size), sigma, border)
    #imgA = cv2.sepFilter2D(img, depth, X, np.transpose(gaussian), delta = delt, borderType = border)
    #imgB = cv2.sepFilter2D(img, depth, gaussian, Y, delta = delt, borderType = border)
    #imgC = cv2.sepFilter2D(blur, depth, X, Y, delta = delt, borderType = border)
    imgA = separable_filter(img, X, np.transpose(gaussian), border, depth, delt)
    imgB = separable_filter(img, gaussian, Y, border, depth, delt)
    imgC = separable_filter(blur, X, Y, border, depth, delt)
    return imgA, imgB, imgC

"""
	Codigo Ejercicios Obligatorios
"""

def ejercicio_1A(img_uri):
    img = readImage(img_uri, 1)
    displayImage(img, name = 'Original')
    plotMultipleImage([gaussian_blur(img, (1, 1), 1), gaussian_blur(img, (11, 11), 1), gaussian_blur(img, (101, 101), 1)], 
                       ['1 - (1, 1)', '1 - (11, 11)', '1 - (101, 101)'], 1, 3, 'GaussianBlur')
    plotMultipleImage([gaussian_blur(img, (1, 1), 3), gaussian_blur(img, (11, 11), 3), gaussian_blur(img, (101, 101), 3)], 
                       ['3 - (1, 1)', '3 - (11, 11)', '3 - (101, 101)'], 1, 3, 'GaussianBlur')
    plotMultipleImage([gaussian_blur(img, (1, 1), 5), gaussian_blur(img, (11, 11), 5), gaussian_blur(img, (101, 101), 5)], 
                       ['5 - (1, 1)', '5 - (11, 11)', '5 - (101, 101)'], 1, 3, 'GaussianBlur')
    plotMultipleImage([gaussian_blur(img, (1, 1), 7), gaussian_blur(img, (11, 11), 7), gaussian_blur(img, (101, 101), 7)], 
                       ['7 - (1, 1)', '7 - (11, 11)', '7 - (101, 101)'], 1, 3, 'GaussianBlur')
	
def ejercicio_1B():
    X1, Y1 = derive_convolution(1, 1, 1)
    X2, Y2 = derive_convolution(2, 2, 1)
    plotMultipleGraph([X1, Y1], ['1º Derivada', 'Size = 1'], 1, 2)
    plotMultipleGraph([X2, Y2], ['2º Derivada', 'Size = 1'], 1, 2)
    X1, Y1 = derive_convolution(1, 1, 3)
    X2, Y2 = derive_convolution(2, 2, 3)
    plotMultipleGraph([X1, Y1], ['1º Derivada', 'Size = 3'], 1, 2)
    plotMultipleGraph([X2, Y2], ['2º Derivada', 'Size = 3'], 1, 2)
    X1, Y1 = derive_convolution(1, 1, 5)
    X2, Y2 = derive_convolution(2, 2, 5)
    plotMultipleGraph([X1, Y1], ['1º Derivada', 'Size = 5'], 1, 2)
    plotMultipleGraph([X2, Y2], ['2º Derivada', 'Size = 5'], 1, 2)
    X1, Y1 = derive_convolution(1, 1, 7)
    X2, Y2 = derive_convolution(2, 2, 7)
    plotMultipleGraph([X1, Y1], ['1º Derivada', 'Size = 7'], 1, 2)
    plotMultipleGraph([X2, Y2], ['2º Derivada', 'Size = 7'], 1, 2)

def ejercicio_1C(img_uri):
    img = readImage(img_uri, 1)
    displayImage(img, name = 'Original')
    plotMultipleImage([laplacian_gaussian(img, 1, cv2.BORDER_REPLICATE), laplacian_gaussian(img, 1, cv2.BORDER_REFLECT)], 
                       ['1 - Replicate', '1 - Reflect'], 1, 3, 'Laplacian')
    plotMultipleImage([laplacian_gaussian(img, 3, cv2.BORDER_REPLICATE), laplacian_gaussian(img, 3, cv2.BORDER_REFLECT)], 
                       ['3 - Replicate', '3 - Reflect'], 1, 3, 'Laplacian')

def ejercicio_2A(img_uri):
    displayImage(separable_filter(gaussian_blur(readImage(img_uri, 0), (11, 11)), 1, 1))    

def ejercicio_2B():
	print()
    #Lo anterior con Derivacion

def ejercicio_2C():
	print()
    #Lo anterior con Derivacion

def ejercicio_2D():
	print()

def ejercicio_2E():
	print()

def ejercicio_3(imgA, imgB, hFreq = 35, lFreq = 99):
    imgA = readImage(imgA, 0)
    imgB = readImage(imgB, 0)
    blurH = gaussian_blur(imgA, (hFreq, hFreq))
    blurL = gaussian_blur(imgB, (lFreq, lFreq))
    sharp = imgA - blurH
    displayMultipleImage([imgA, sharp+blurL, imgB])

"""
	Codigo Ejercicios Bonus
"""

def ejercicio_B1():
	print()

def ejercicio_B2():
	print()

def ejercicio_B3():
	print()

def ejercicio_B4():
	print()

def ejercicio_B5():
	print()
	
"""
	Codigo Principal
"""

def main():
	### Ejercicio 01 - A
    #ejercicio_1A('data/cat.bmp')
    #input("\nPulsa Enter para continuar la ejecucion:\n")
	### Ejercicio 01 - B
    #ejercicio_1B()
    #input("\nPulsa Enter para continuar la ejecucion:\n")
    ### Ejercicio 01 - C
    #ejercicio_1C('data/dog.bmp')
    #input("\nPulsa Enter para continuar la ejecucion:\n")
    ### Ejercicio 02 - A
    #ejercicio_2A('data/plane.bmp')
    #input("\nPulsa Enter para continuar la ejecucion:\n")
    ### Ejercicio 03
    #ejercicio_3('data/cat.bmp', 'data/dog.bmp')
    #input("\nPulsa Enter para continuar la ejecucion:\n")
    
if __name__ == "__main__":
	main()