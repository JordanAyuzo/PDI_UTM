import cv2
import matplotlib.pyplot as plt
import numpy as np

def mostrar_menu_filtros():
    print("\nSelecciona el filtro a utilizar:")
    print("1. Ideal Low Pass Filter")
    print("2. Butterworth Low Pass Filter")
    print("3. Gaussian Low Pass Filter")
    print("4. Ideal High Pass Filter")
    print("5. Butterworth High Pass Filter")
    print("6. Gaussian High Pass Filter")
    print("0. Salir")
    
    opcion = int(input("\nOpcion: "))
    return opcion

def mostrar_menu_imagenes():
    print("\nSelecciona la imagen a utilizar:")
    print("1. Imagen Alto Contraste")
    print("2. Imagen Bajo Contraste")
    print("3. Imagen Alta Iluminacion")
    print("4. Imagen Baja Iluminacion")
    
    opcion = int(input("\nOpcion: "))
    return opcion

def cargar_imagen(ruta):
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("Error al cargar la imagen.")
        return None
    return img

def ideal_low_pass_filter(width, height, d):
    lp_filter = np.zeros((height, width), np.float32)
    centre = (width // 2, height // 2)

    for i in range(height):
        for j in range(width):
            radius = np.sqrt((i - centre[1])**2 + (j - centre[0])**2)
            lp_filter[i, j] = 1 if radius <= d else 0
    return lp_filter

def butterworth_low_pass_filter(width, height, d, n):
    lp_filter = np.zeros((height, width), np.float32)
    centre = (width // 2, height // 2)

    for i in range(height):
        for j in range(width):
            radius = np.sqrt((i - centre[1])**2 + (j - centre[0])**2)
            lp_filter[i, j] = 1 / (1 + (radius / d)**(2 * n))
    return lp_filter

def gaussian_low_pass_filter(width, height, d):
    lp_filter = np.zeros((height, width), np.float32)
    centre = (width // 2, height // 2)

    for i in range(height):
        for j in range(width):
            radius = np.sqrt((i - centre[1])**2 + (j - centre[0])**2)
            lp_filter[i, j] = np.exp(-(radius**2) / (2 * d**2))
    return lp_filter

def ideal_high_pass_filter(width, height, d):
    return 1 - ideal_low_pass_filter(width, height, d)

def butterworth_high_pass_filter(width, height, d, n):
    lp_filter = butterworth_low_pass_filter(width, height, d, n)
    return 1 - lp_filter

def gaussian_high_pass_filter(width, height, d):
    lp_filter = gaussian_low_pass_filter(width, height, d)
    return 1 - lp_filter

def aplicar_filtro(img, filtro, d, n=None):
    height, width = img.shape
    if filtro == "Ideal Low Pass Filter":
        mask = ideal_low_pass_filter(width, height, d)
    elif filtro == "Butterworth Low Pass Filter":
        mask = butterworth_low_pass_filter(width, height, d, n)
    elif filtro == "Gaussian Low Pass Filter":
        mask = gaussian_low_pass_filter(width, height, d)
    elif filtro == "Ideal High Pass Filter":
        mask = ideal_high_pass_filter(width, height, d)
    elif filtro == "Butterworth High Pass Filter":
        mask = butterworth_high_pass_filter(width, height, d, n)
    elif filtro == "Gaussian High Pass Filter":
        mask = gaussian_high_pass_filter(width, height, d)
    else:
        print("Filtro no implementado.")
        return None

    dft = np.fft.fft2(img)
    dft_shift = np.fft.fftshift(dft)
    filtered_dft = dft_shift * mask
    dft_inv = np.fft.ifftshift(filtered_dft)
    img_filtered = np.fft.ifft2(dft_inv).real

    return img_filtered

def mostrar_imagenes(img_original, img_filtrada, filtro):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Imagen Original")
    plt.imshow(img_original, cmap='gray')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.title(f"{filtro}")
    plt.imshow(img_filtrada, cmap='gray')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    paths = {
        1: "codigos/../imagenes/AC.jpg",
        2: "codigos/../imagenes/BC.jpg",
        3: "codigos/../imagenes/AI.jpg",
        4: "codigos/../imagenes/BI.jpg"
    }
    filtros = {
        1: "Ideal Low Pass Filter",
        2: "Butterworth Low Pass Filter",
        3: "Gaussian Low Pass Filter",
        4: "Ideal High Pass Filter",
        5: "Butterworth High Pass Filter",
        6: "Gaussian High Pass Filter"
    }

    while True:
        filtro_opcion = mostrar_menu_filtros()
        if filtro_opcion == 0:
            print("Saliendo del programa...")
            break
        elif filtro_opcion not in filtros:
            print("\nOpcion invalida. Intenta de nuevo.")
            continue
        img_opcion = mostrar_menu_imagenes()
        if img_opcion not in paths:
            print("\nOpcion invalida. Intenta de nuevo.")
            continue
        img = cargar_imagen(paths[img_opcion])
        if img is not None:
            d = int(input("Ingresa el radio de corte: "))
            n = None
            if filtro_opcion in [2, 5]:
                n = int(input("Ingresa el orden del filtro: "))
            filtro = filtros[filtro_opcion]
            img_filtrada = aplicar_filtro(img, filtro, d, n)
            if img_filtrada is not None:
                mostrar_imagenes(img, img_filtrada, filtro)

if __name__ == "__main__":
    main()
