import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
#Transformaciones
def normalizar(imagen):
    return np.uint8(255 * (imagen / np.max(imagen)))

def negativo(imagen):
    return 255 - imagen

def logaritmica(imagen, c=1):
    imagen_float = np.float32(imagen)  
    log_imagen = c * np.log1p(imagen_float) 
    log_imagen_normalizada = cv2.normalize(log_imagen, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(log_imagen_normalizada)

def gamma_correction(imagen, gamma, c=1):
    return normalizar( c * (imagen ** (gamma)))

def estiramiento_contraste(imagen):
    min_val = np.min(imagen)
    max_val = np.max(imagen)
    imagen_estirada = (imagen - min_val) * (255 / (max_val - min_val))
    imagen_estirada = np.clip(imagen_estirada, 0, 255).astype(np.uint8)

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

#Procesamiento de la imagen

def mostrar(nombre,img):
    cv2.imshow(nombre,img) 
    cv2.waitKey(0) 
    #cv2.destroyAllWindows()

def comoes(img):
    print("Tamaño:" , img.shape)#devuelve pixel,pixel,canales
    print("  max :" , np.max(img))#el numero maximo 
    print("  max :" , np.min(img))#devuelve el minimo de la matriz

def negativos(altoContraste,bajoContraste,pocaIluminacion):
    output_dir = 'transformadas/negativos'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    negativo1 = negativo(altoContraste)
    output_path = os.path.join(output_dir, 'Alto-Contraste.jpg')
    cv2.imwrite(output_path, negativo1)
    negativo2 = negativo(bajoContraste)
    output_path = os.path.join(output_dir, 'Bajo-Contraste.jpg')
    cv2.imwrite(output_path, negativo2)
    negativo3 = negativo(pocaIluminacion)
    output_path = os.path.join(output_dir, 'Baja-Iluminacion.jpg')
    cv2.imwrite(output_path, negativo3)

def logaritmicos(altoContraste,bajoContraste,pocaIluminacion):
    output_dir = 'transformadas/logaritmica'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    log1 = logaritmica(altoContraste)
    output_path = os.path.join(output_dir, 'Alto-Contraste.jpg')
    cv2.imwrite(output_path, log1)
    log2 = logaritmica(bajoContraste)
    output_path = os.path.join(output_dir, 'Bajo-Contraste.jpg')
    cv2.imwrite(output_path, log2)
    log3 = logaritmica(pocaIluminacion)
    output_path = os.path.join(output_dir, 'Poca-Iluminacion.jpg')
    cv2.imwrite(output_path, log3)

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

def gamma(img,tipo):
    output_dir = 'transformadas/gamma/'+ tipo
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    for i in range(0,11):
        gamma_img = gamma_correction(img,i*0.1, 1)
        output_path = os.path.join(output_dir, 'gamma_'+str(round(i * 0.1, 1))+'.jpg')
        cv2.imwrite(output_path, gamma_img)

def estiramiento(altoContraste,bajoContraste,pocaIluminacion):
    #impirmir los primeros 5 valores de la imagen
    mostrar_histograma(bajoContraste)
    output_dir = 'transformadas/estiramiento'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img1 = estiramiento_contraste(altoContraste)
    output_path = os.path.join(output_dir, 'Alto-Contraste.jpg')
    cv2.imwrite(output_path, img1)
    img2 = estiramiento_contraste(bajoContraste)
    output_path = os.path.join(output_dir, 'Bajo-Contraste.jpg')
    cv2.imwrite(output_path, img2)
    mostrar_histograma(img2)
    img3 = estiramiento_contraste(pocaIluminacion)
    output_path = os.path.join(output_dir, 'Poca-Iluminacion.jpg')
    cv2.imwrite(output_path, img3)
    
def intensity_planes(altoContraste,bajoContraste,pocaIluminacion):
    #impirmir los primeros 5 valores de la imagen
    output_dir = 'transformadas/intensity_planes'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    img1 = rebanada_intensidad(altoContraste, 20, 30)
    output_path = os.path.join(output_dir, 'Alto-Contraste.jpg')
    cv2.imwrite(output_path, img1)
    img2 = rebanada_intensidad(bajoContraste, 180, 200)
    output_path = os.path.join(output_dir, 'Bajo-Contraste.jpg')
    cv2.imwrite(output_path, img2)
    img3 = rebanada_intensidad(pocaIluminacion, 10, 30)
    output_path = os.path.join(output_dir, 'Poca-Iluminacion.jpg')
    cv2.imwrite(output_path, img3)
    
def mostrar_histograma(imagen):
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
def main():
    img1 = cv2.imread('codigos/../imagenes/img1-AltoContraste.jpg',0)
    img2 = cv2.imread('codigos/../imagenes/img1-BajoContraste.jpg',0)
    img3 = cv2.imread('codigos/../imagenes/img1-PocaIluminacion.jpg',0)
    #Negativo
    negativos(img1,img2,img3)
    
    #logaritmica
    logaritmicos(img1,img2,img3)
    
    #gamma
    gamma(img1,"Alto-Contraste")
    gamma(img2,"Bajo-Contraste")
    gamma(img3,"Poca-Iluminacion")
    
    #Rebanada de plano de bits
    bit_planes(img1)
    bit_planes(img2)
    bit_planes(img3)
    
    #Estiramiento por contraste
    estiramiento(img1,img2,img3)
    
    #Rebanada de intensidad

    intensity_planes(img1,img2,img3)
main()
