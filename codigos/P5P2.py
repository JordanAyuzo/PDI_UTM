import numpy as np
import matplotlib.pyplot as plt
import cv2

#Menu de colores
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
BRIGHT_GREEN = "\033[92m"
BRIGHT_YELLOW = "\033[93m"
BRIGHT_BLUE = "\033[94m"
BRIGHT_MAGENTA = "\033[95m"
RESET = "\033[0m"  # Para restablecer el color
RED = "\033[31m"
def filtro_gradiente(imagen):
    # Gradiente se calcula con Sobel (dirección X e Y)
    grad_x = cv2.Sobel(imagen, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(imagen, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)

def filtro_laplaciano(imagen):
    return cv2.Laplacian(imagen, cv2.CV_64F)

def filtro_minimo(imagen, tamaño_kernel=3):
    return cv2.erode(imagen, np.ones((tamaño_kernel, tamaño_kernel), np.uint8))

def filtro_maximo(imagen, tamaño_kernel=3):
    return cv2.dilate(imagen, np.ones((tamaño_kernel, tamaño_kernel), np.uint8))

def filtro_mediana(imagen, tamaño_kernel=3):
    return cv2.medianBlur(imagen, tamaño_kernel)

def filtro_promedio(imagen, tamaño_kernel=3):
    kernel = np.ones((tamaño_kernel, tamaño_kernel), np.float32) / (tamaño_kernel**2)
    return cv2.filter2D(imagen, -1, kernel)

def mostrar_comparacion(imagen1, imagen2, titulo1="Imagen 1", titulo2="Imagen 2", cmap="gray"):
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(imagen1, cmap=cmap)
    plt.title(titulo1)
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.imshow(imagen2, cmap=cmap)
    plt.title(titulo2)
    plt.axis("off")
    
    plt.tight_layout()
    plt.show()

def mostrarImagen(img,nombre):
    plt.imshow(img, cmap='gray')
    plt.title(nombre)
    plt.show()

def MenuImagen(path, titulo):
    img = cv2.imread(path,0)
    if img is None:
        print(f"{RED}Error: No se pudo leer la imagen. Asegúrese de que la imagen exista y la ruta sea correcta.{RESET}")
        print(f"{YELLOW}¨SUGERENCIA: Dirección correcta de la imagen: PDI_UTM/imagenes/[TU IMAGEN].{RESET}")
        print(f"{YELLOW}¨SUGERENCIA: Si elegiste la opción de imagen personalizada solo es el nombre Ej: img.jpg.{RESET}")
        input("Presione cualquier tecla para continuar...")
        return  
    while True:
        print(f"{BLUE}Imagen: {titulo}{RESET}")
        print("---- Menú -----")
        print (f"{GREEN}1.- Mostrar Imagen{RESET}")
        print (f"{YELLOW}2.- Aplicar Filtro Promedio{RESET}")
        print (f"{MAGENTA}3.- Aplicar Filtro Mediana{RESET}")
        print (f"{BRIGHT_GREEN}4.- Aplicar Filtro Máximo{RESET}")
        print (f"{BRIGHT_YELLOW}5.- Aplicar Filtro Mínimo{RESET}")
        print (f"{BRIGHT_MAGENTA}6.- Aplicar Filtro Laplaciano{RESET}")
        print (f"{BRIGHT_BLUE}7.- Aplicar Filtro Gradiente{RESET}")
        print (f"{RED}8.- Salir{RESET}")
        opc = int(input())
        print("\033[H\033[J")
        if opc == 1:
            mostrarImagen(img, titulo)
        if opc == 2:
            print (f"{YELLOW} --- Filtro Promedio ---{RESET}")
            kernel = int(input(" > Ingrese el tamaño de la mascara: "))
            img_procesada = filtro_promedio(img,kernel)
            mostrar_comparacion(img, img_procesada, "Original", "Filtro Promedio")
        elif opc == 3:
            print (f"{MAGENTA} --- Filtro Mediana ---{RESET}")
            kernel = int(input(" > Ingrese el tamaño de la mascara: "))
            img_procesada = filtro_mediana(img,kernel)
            mostrar_comparacion(img, img_procesada, "Original", "Filtro Mediana")
        elif opc == 4:
            print (f"{BRIGHT_GREEN} --- Filtro Máximo ---{RESET}")
            kernel = int(input(" > Ingrese el tamaño de la mascara: "))
            img_procesada = filtro_maximo(img,kernel)
            mostrar_comparacion(img, img_procesada, "Original", "Filtro Máximo")
        elif opc == 5:
            print (f"{BRIGHT_YELLOW} --- Filtro Mínimo ---{RESET}")
            kernel = int(input(" > Ingrese el tamaño de la mascara: "))
            img_procesada = filtro_minimo(img,kernel)
            mostrar_comparacion(img, img_procesada, "Original", "Filtro Mínimo")
        elif opc == 6:
            print(f"{BRIGHT_MAGENTA} --- Filtro Laplaciano ---{RESET}")
            
            # Aplicar filtro Laplaciano
            img_laplaciana = filtro_laplaciano(img)
            
            # Convertir el resultado del Laplaciano al rango y tipo de la imagen original
            img_laplaciana = cv2.convertScaleAbs(img_laplaciana)  # Convierte a uint8 (0-255)
            
            # Sumar la imagen original con el Laplaciano
            img_procesada = cv2.add(img, img_laplaciana)
            
            # Mostrar comparación
            mostrar_comparacion(img, img_procesada, "Original", "Original + Laplaciano")
        elif opc == 7:
            print(f"{BRIGHT_BLUE} --- Filtro Gradiente ---{RESET}")
        
            # Aplicar filtro de gradiente
            img_gradiente = filtro_gradiente(img)
            
            # Convertir el gradiente al rango y tipo de la imagen original
            img_gradiente = cv2.convertScaleAbs(img_gradiente)  # Asegura valores en rango [0, 255]
            
            # Sumar la imagen original con el gradiente
            img_procesada = cv2.add(img, img_gradiente)
            
            # Mostrar comparación
            mostrar_comparacion(img, img_procesada, "Original", "Original + Gradiente")
        elif opc == 8:
            break
        print("\033[H\033[J")
def main():
    # Limpiar la consola
    print("\033[H\033[J")
    #Menu Principal
    while True:
        print(f"{BLUE}Elige una imagen para calcular su histograma{RESET}")
        print(f"{GREEN}1.- Alto Contraste{RESET}")
        print(f"{YELLOW}2.- Bajo Contraste{RESET}")
        print(f"{MAGENTA}3.- Alta Iluminacion{RESET}")
        print(f"{BRIGHT_BLUE}4.- Baja Iluminacion{RESET}")
        print(f"{BRIGHT_MAGENTA}5.- Sal y Pimienta{RESET}")
        print(f"{BRIGHT_GREEN}6.- Imagen Personalizada{RESET}")
        print(f"{RED}7.- Salir{RESET}")
        
        opc = int(input())
        print("\033[H\033[J")
        if opc == 1:
            MenuImagen('codigos/../imagenes/AC.jpg', 'Alto Contraste')
        elif opc == 2:
            MenuImagen('codigos/../imagenes/BC.jpg', 'Bajo Contraste')
        elif opc == 3:
            MenuImagen('codigos/../imagenes/AI.jpg', 'Alta Iluminacion')
        elif opc == 4:
            MenuImagen('codigos/../imagenes/BI.jpg', 'Baja Iluminacion')
        elif opc == 5:
            MenuImagen('codigos/../imagenes/SyP.jpeg', 'Sal y Pimienta')
        elif opc == 6:
            print("Agrega la imagen en la carpeta imagenes y escribe el nombre de la imagen")
            path = input("Nombre de la imagen: ")
            MenuImagen('codigos/../imagenes/'+path, 'Imagen Personalizada') 
        elif opc == 7:
            break
        else:
            print("Opcion no valida")

main()