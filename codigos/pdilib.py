import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import inspect

#Manipulacion una imagen

def importar(ruta, capa):
    return cv2.imread(ruta, capa)

def mostrarCV2(nombre, img):
    cv2.imshow(nombre, img)  # muestra la imagen en pantalla
    cv2.waitKey(0)  # espera a que presiones cualquier botón
    cv2.destroyAllWindows()  # opcional: cierra todas las ventanas que tengas abiertas de cv

def mostrarPlt(nombre, img):
    plt.imshow(img, cmap='gray')
    plt.title(nombre)
    plt.show()

def guardarImagen(nombre, ruta, imagen):
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    output_path = os.path.join(ruta, nombre)  # Ruta de salida
    cv2.imwrite(output_path, imagen)

def redimensionarCuadrado(img, tam):
    return cv2.resize(img, (tam, tam))  # Cambia para que el tamaño sea cuadrado

def redemiencionarProporcional(img, ancho):
    return cv2.resize(img, (ancho, int(img.shape[0] * ancho / img.shape[1])))  # Cambia para que el tamaño sea cuadrado

def normalizar(imagen):
    return np.uint8(255 * (imagen / np.max(imagen)))

def histograma(imagen):
    # Calcular el histograma usando OpenCV
    histograma = cv2.calcHist([imagen], [0], None, [256], [0, 256])
    histograma = histograma / np.sum(histograma)
    # Graficar el histograma
    plt.figure(figsize=(10, 6))
    plt.plot(histograma, color='black')
    plt.title('Histograma de la imagen en escala de grises')
    plt.xlabel('Valores de intensidad')
    plt.ylabel('Frecuencia')
    plt.grid(True)
    plt.xlim([0, 256])  # El rango de intensidades es de 0 a 255
    plt.show()
#Operaciones de bits

def operador_not(img):
    return cv2.bitwise_not(img)

def operador_and(img1, img2):
    return cv2.bitwise_and(img1, img2)

def operador_or(img1, img2):
    return cv2.bitwise_or(img1, img2)

def operador_nand(img1, img2):
    and_img = cv2.bitwise_and(img1, img2)
    return cv2.bitwise_not(and_img)

def operador_xor(img1, img2):
    return cv2.bitwise_xor(img1, img2)

#Transformaciones

def negativo(imagen):
    return 255 - imagen

def logaritmica(imagen, c=1):
    imagen_float = np.float32(imagen)  
    log_imagen = c * np.log1p(imagen_float) 
    log_imagen_normalizada = cv2.normalize(log_imagen, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(log_imagen_normalizada)

def gamma_correction(imagen, gamma, c=1):
    return normalizar( c * (imagen ** (gamma)))

def estiramiento_contraste(imagen, min_out=0, max_out=255, min_in=0, max_in=255):
    return (imagen - min_in) * ((max_out - min_out) / (max_in - min_in)) + min_out

    return imagen_estirada

def rebanada_intensidad_V2(imagen, min_val, max_val, nuevo_val=0):
    # Copiar la imagen original para que los valores dentro del rango permanezcan iguales
    salida = np.full_like(imagen, nuevo_val)  # Inicializar la salida con nuevo_val
    
    # Mantener los valores originales dentro del rango [min_val, max_val]
    salida[(imagen >= min_val) & (imagen <= max_val)] = imagen[(imagen >= min_val) & (imagen <= max_val)]
    
    return salida

def rebanada_intensidad(imagen, min_val, max_val, nuevo_val=255):
    salida = np.zeros_like(imagen)
    salida[(imagen >= min_val) & (imagen <= max_val)] = nuevo_val
    return salida

def rebanada_plano_bit(imagen, bit_plano):
    return (imagen >> bit_plano) & 1

def bit_planes(img):
    if img is None:
        print("Error al cargar la imagen")
    else:
        rows, cols = img.shape
        bit_planes = []
        for i in range(8):
            bit_plane = rebanada_plano_bit(img,i) #desplazamiento de bits
            bit_plane = bit_plane * 255
            bit_planes.append(bit_plane)

        # Mostrar los planos de bits utilizando matplotlib
        fig, axs = plt.subplots(2, 4, figsize=(12, 6))
        for i in range(8):
            ax = axs[i // 4, i % 4]
            ax.imshow(bit_planes[i], cmap='gray')
            ax.set_title(f'Bit plane {i}')
            ax.axis('off')
        plt.tight_layout()
        plt.show()

def helpFunction(funcion):
    print(inspect.getsource(funcion))
    
def help():
    #imprimir que funciones existen
    print("Funciones disponibles:")
    print("importar(ruta, capa)")
    print("mostrarCV2(nombre, img)")
    print("mostrarPlt(nombre, img)")
    print("guardarImagen(nombre, ruta, imagen)")
    print("operador_not(img)")
    print("operador_and(img1, img2)")
    print("operador_or(img1, img2)")
    print("operador_nand(img1, img2)")
    print("operador_xor(img1, img2)")
    print("redimensionarCuadrado(img, tam)")
    print("redemiencionarProporcional(img, ancho)")
    print("normalizar(imagen)")
    print("negativo(imagen)")
    print("logaritmica(imagen, c=1)")
    print("gamma_correction(imagen, gamma, c=1)")
    print("estiramiento_contraste(imagen, min_out=0, max_out=255, min_in=0, max_in=255)")
    print("rebanada_intensidad_V2(imagen, min_val, max_val, nuevo_val=0)")
    print("rebanada_intensidad(imagen, min_val, max_val, nuevo_val=255)")
    print("rebanada_plano_bit(imagen, bit_plano)")
    print("bit_planes(img)")
    print("help()")
    print("histograma(imagen)")
    print("helpFunction(funcion)")
    print("main()")