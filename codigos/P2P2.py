import numpy as np
import matplotlib.pyplot as plt
import cv2

#Menu de colores
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
RESET = "\033[0m"  # Para restablecer el color


def mostrarImagen(img,nombre):
    plt.imshow(img, cmap='gray')
    plt.title(nombre)
    plt.show()

def comparacion(imagen1, histograma1, titulo1, imagen2, histograma2, titulo2, imagen3=None, histograma3=None, titulo3=None):
    if imagen3 is not None and histograma3 is not None:
        # Crear una figura con 2 filas y 3 columnas para tres imágenes y sus histogramas
        fig, axs = plt.subplots(2, 3, figsize=(14, 6))
    else:
        # Crear una figura con 2 filas y 2 columnas para dos imágenes y sus histogramas
        fig, axs = plt.subplots(2, 2, figsize=(12, 6))

    # Mostrar la primera imagen y su histograma
    axs[0, 0].imshow(imagen1, cmap='gray')
    axs[0, 0].set_title(titulo1)
    axs[0, 0].axis('off')
    
    axs[1, 0].bar(range(256), histograma1, width=1.0, color='gray')
    axs[1, 0].set_title("Histograma - " + titulo1)
    axs[1, 0].set_xlabel("Intensidad de píxel")
    axs[1, 0].set_ylabel("Frecuencia")
    axs[1, 0].grid(True)
    
    # Mostrar la segunda imagen y su histograma
    axs[0, 1].imshow(imagen2, cmap='gray')
    axs[0, 1].set_title(titulo2)
    axs[0, 1].axis('off')
    
    axs[1, 1].bar(range(256), histograma2, width=1.0, color='gray')
    axs[1, 1].set_title("Histograma - " + titulo2)
    axs[1, 1].set_xlabel("Intensidad de píxel")
    axs[1, 1].set_ylabel("Frecuencia")
    axs[1, 1].grid(True)
    
    # Mostrar la tercera imagen y su histograma si existen
    if imagen3 is not None and histograma3 is not None:
        axs[0, 2].imshow(imagen3, cmap='gray')
        axs[0, 2].set_title(titulo3)
        axs[0, 2].axis('off')
        
        axs[1, 2].bar(range(256), histograma3, width=1.0, color='gray')
        axs[1, 2].set_title("Histograma - " + titulo3)
        axs[1, 2].set_xlabel("Intensidad de píxel")
        axs[1, 2].set_ylabel("Frecuencia")
        axs[1, 2].grid(True)
    
    # Ajustar el espacio entre subplots
    plt.tight_layout()
    plt.show()

def mostrarHistograma(histograma1, histograma2, L=256, titulo="Histograma de la Imagen", titulo1="Histograma 1", titulo2="Histograma 2"):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(titulo)
    
    # Primer histograma
    axs[0].bar(range(L), histograma1, width=1.0, color='gray')
    axs[0].set_title(titulo1)
    axs[0].set_xlabel("Intensidad de píxel")
    axs[0].set_ylabel("Frecuencia")
    axs[0].grid(True)
    
    # Segundo histograma
    axs[1].bar(range(L), histograma2, width=1.0, color='gray')
    axs[1].set_title(titulo2)
    axs[1].set_xlabel("Intensidad de píxel")
    axs[1].set_ylabel("Frecuencia")
    axs[1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
def histograma(imagen):
    histograma = np.zeros(256, dtype=int)
    for valor in imagen.flatten():
        histograma[valor] += 1
    return histograma

def histogramaNormalizado(imagen):
    hist = histograma(imagen)

    total_pixeles = imagen.size
    histograma_normalizado = hist / total_pixeles
    
    return histograma_normalizado

def global_histogram_equalization(image):
    # Aplica ecualización de histograma global
    equalized_image = cv2.equalizeHist(image)
    return equalized_image

def local_histogram_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Verifica si la imagen está en escala de grises
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Crea el objeto CLAHE con los parámetros de límite y tamaño de cuadricula
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    local_equalized_image = clahe.apply(image)
    return local_equalized_image

def estadisticas_globales(imagen):
    media_global = np.mean(imagen)
    varianza_global = np.var(imagen)
    return media_global, varianza_global

def estadisticas_locales(imagen, bloque=8):
    m, n = imagen.shape
    offset = bloque // 2
    
    # Crear matrices para almacenar la media y varianza locales
    media_local = np.zeros_like(imagen, dtype=float)
    varianza_local = np.zeros_like(imagen, dtype=float)
    
    # Calcular media y varianza en cada vecindad
    for i in range(offset, m - offset):
        for j in range(offset, n - offset):
            # Definir la vecindad centrada en el píxel (i, j)
            vecindad = imagen[i - offset:i + offset + 1, j - offset:j + offset + 1]
            
            # Calcular la media y varianza de la vecindad
            media = np.mean(vecindad)
            varianza = np.var(vecindad)
            
            # Asignar los valores a la posición correspondiente
            media_local[i, j] = media
            varianza_local[i, j] = varianza
    
    return media_local, varianza_local

def procesamiento_global(imagen, medida, umbral_varianza=500):
    # Aplicar ecualización de histograma global si la varianza global es baja
    if medida < umbral_varianza:
        print("Aplicando ecualización de histograma global.")
        return global_histogram_equalization(imagen)
    else:
        print("La varianza es alta. No se necesita ecualización global.")
        return imagen  # Retorna la imagen sin cambios si no requiere ecualización

def local_media(imagen, umbral_varianza=500, bloque=8):
    # Calcular estadísticas locales
    media,_  = estadisticas_locales(imagen, bloque)

    # Definir una máscara donde aplicar ecualización local (alta varianza local)
    mascara_alta_varianza = media > umbral_varianza

    # Aplicar ecualización de histograma local en vecindades donde la varianza es alta
    print("Aplicando ecualización de histograma local en zonas de alta varianza.")
    return local_histogram_equalization(imagen, bloque)


def local_varianza(imagen, umbral_varianza=500, bloque=8):
    # Calcular estadísticas locales
    _, varianza_local = estadisticas_locales(imagen, bloque)

    # Definir una máscara donde aplicar ecualización local (alta varianza local)
    mascara_alta_varianza = varianza_local > umbral_varianza

    # Aplicar ecualización de histograma local en vecindades donde la varianza es alta
    print("Aplicando ecualización de histograma local en zonas de alta varianza.")
    return local_histogram_equalization(imagen, bloque)


def MenuVarianza(img, titulo="Imagen"):
    media_global, varianza_global = estadisticas_globales(img)
    print(f"Media Global: {media_global}")
    print(f"Varianza Global: {varianza_global}")
    while True:
        print(f"{BLUE}Imagen: {titulo}{RESET}")
        print("Menu:")
        print (f"{GREEN}Media global:{media_global} Varianza Local:{varianza_global} {RESET}")
        print (f"{YELLOW}1.- Procesamiento Media global{RESET}")
        print (f"{RED}2.- Procesamiento Varianza global{RESET}")
        print (f"{YELLOW}3.- Procesamiento Media Local{RESET}")
        print (f"{RED}4.- Procesamiento Varianza Local{RESET}")
        print (f"{YELLOW}5.- Salir{RESET}")
        opc = int(input())
        print("\033[H\033[J")
        if opc == 1:
            img_procesada = procesamiento_global(img, media_global,)
            histograma_img = histograma(img)
            comparacion(img, histograma_img, "Original", img_procesada, histograma(img_procesada), "Media Global")
        elif opc == 2:
            img_procesada = procesamiento_global(img, varianza_global)
            histograma_img = histograma(img)
            comparacion(img, histograma_img, "Original", img_procesada, histograma(img_procesada), "Varianza Global")
        elif opc == 3:
            humbral = int(input("Valor de humbral:"))
            bloque = int(input("valor del bloque: "))
            img_procesada = local_media(img, humbral,bloque)
            histograma_img = histograma(img)
            comparacion(img, histograma_img, "Original", img_procesada, histograma(img_procesada), "Media Global")
        elif opc == 4:
            humbral = int(input("Valor de humbral:"))
            bloque = int(input("valor del bloque: "))
            img_procesada = local_varianza(img, humbral,bloque)
            histograma_img = histograma(img)
            comparacion(img, histograma_img, "Original", img_procesada, histograma(img_procesada), "Media Global")
        elif opc == 5:
            break   

def MenuImagen(path, titulo="Imagen"):
    img = cv2.imread(path,0)
    while True:
        print(f"{BLUE}Imagen: {titulo}{RESET}")
        print("Menu:")
        print (f"{GREEN}1.- Mostrar Imagen{RESET}")
        print (f"{YELLOW}2.- Mostrar Histograma(Original y Normalizado){RESET}")
        print (f"{RED}3.- Menu de Media y varianza(Global y Local){RESET}")
        print (f"{MAGENTA}4.- Ecualizar Histograma (Global y Local){RESET}")
        print (f"{BLUE}5.- Ecualizar histograma Global{RESET}")
        print (f"{RED}6.- Ecualizar histograma Local{RESET}")
        print (f"{YELLOW}7.- Salir{RESET}")
        opc = int(input())
        print("\033[H\033[J")
        if opc == 1:
            mostrarImagen(img, titulo)
        elif opc == 2:
            mostrarHistograma(histograma(img),histogramaNormalizado(img), titulo="Histograma de la Imagen", titulo1="Histograma", titulo2="Histograma Normalizado")
        elif opc == 3:
            MenuVarianza(img, titulo)
        elif opc == 4:
            x = int(input("Ingresa el tamaño de la cuadrícula(8): "))
            global_equalized_image = global_histogram_equalization(img)
            global_equalized_histogram = histograma(global_equalized_image)
            local_equalized_image = local_histogram_equalization(img, clip_limit=2.0, tile_grid_size=(x, x))
            local_equalized_histogram = histograma(local_equalized_image)
            comparacion(img, histograma(img), "Original", global_equalized_image, global_equalized_histogram, "Global Equalizada",local_equalized_image, local_equalized_histogram, "Local Equalizada")
        elif opc == 5:
            global_equalized_image = global_histogram_equalization(img)
            global_equalized_histogram = histograma(global_equalized_image)
            comparacion(img, histograma(img), "Original", global_equalized_image, global_equalized_histogram, "Global Equalizada")
        elif opc == 6:
            clip_limit = float(input("Ingresa el límite de recorte(Dejarlo en 2.0): "))
            x = int(input("Ingresa el tamaño de la cuadrícula(8): ")) 
            local_equalized_image = local_histogram_equalization(img, clip_limit=clip_limit, tile_grid_size=(x, x))
            local_equalized_histogram = histograma(local_equalized_image)
            comparacion(img, histograma(img), "Original", local_equalized_image, local_equalized_histogram, "Local Equalizada")
            print("\033[H\033[J")
        elif opc == 7:
            break
        else:
            print("Opcion no valida")

def main():
    # Limpiar la consola
    print("\033[H\033[J")
    
    #Menu Principal
    while True:
        print(f"{BLUE}Elige una imagen para calcular su histograma{RESET}")
        print(f"{GREEN}1.- Alto Contraste{RESET}")
        print(f"{YELLOW}2.- Bajo Contraste{RESET}")
        print(f"{MAGENTA}3.- Alta Iluminacion{RESET}")
        print(f"{MAGENTA}4.- Poca Iluminacion{RESET}")
        print(f"{RED}5.- Imagen Personalizada{RESET}")
        print(f"{RESET}6.- Salir{RESET}")
        
        opc = int(input())
        print("\033[H\033[J")
        if opc == 1:
            MenuImagen('codigos/../imagenes/AC.jpg', 'Alto Contraste')
        elif opc == 2:
            MenuImagen('codigos/../imagenes/BC.jpg', 'Bajo Contraste')
        elif opc == 3:
            MenuImagen('codigos/../imagenes/AI.jpg', 'Alta Iluminacion')
        elif opc == 4:
            MenuImagen('codigos/../imagenes/BI.jpg', 'Poca Iluminacion')
        elif opc == 5:
            print("Agrega la imagen en la carpeta imagenes y escribe el nombre de la imagen")
            path = input("Nombre de la imagen: ")
            MenuImagen('codigos/../imagenes/'+path, 'Imagen Personalizada') 
        elif opc == 6:
            break
        else:
            print("Opcion no valida")

main()