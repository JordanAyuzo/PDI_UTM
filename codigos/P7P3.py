import cv2
import numpy as np
import matplotlib.pyplot as plt

def load_image():
    """Display menu to select an image."""
    print("Selecciona una imagen:")
    print("1. img1")
    print("2. img2")
    print("3. img3")
    choice = input("Ingrese el número de la imagen: ")
    if choice == "1":
        return cv2.imread("codigos/../imagenes/img01.png", 0)
    elif choice == "2":
        return cv2.imread("codigos/../imagenes/img02.png", 0)
    elif choice == "3":
        return cv2.imread("codigos/../imagenes/img03.png", 0)
    else:
        print("Opción inválida. Inténtalo de nuevo.")
        return load_image()

def display_images(original, processed, title="Resultado"):
    """Display the original and processed images side by side using Matplotlib."""
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original")
    plt.imshow(original, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.title("Procesada")
    plt.imshow(processed, cmap="gray")
    plt.axis("off")

    plt.suptitle(title)
    plt.show()

def morphological_operations(img):
    """Perform basic morphological operations based on mathematical definitions."""
    print("Operaciones morfológicas:")
    print("1. Erosión: Reduce objetos eliminando píxeles en los bordes.")
    print("2. Dilatación: Amplía objetos añadiendo píxeles en los bordes.")
    print("3. Apertura: Suaviza contornos eliminando ruido (Erosión seguida de Dilatación).")
    print("4. Cierre: Rellena huecos pequeños (Dilatación seguida de Erosión).")
    choice = input("Selecciona una operación: ")

    kernel_size = int(input("Introduce el tamaño del kernel (e.g., 3): "))
    print("Selecciona la forma del elemento estructurante:")
    print("1. Rectangular")
    print("2. Elíptico")
    print("3. Cruz")
    shape_choice = input("Introduce el número de la forma: ")

    if shape_choice == "1":
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
    elif shape_choice == "2":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    elif shape_choice == "3":
        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))
    else:
        print("Opción inválida. Usando forma rectangular por defecto.")
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))

    if choice == "1":
        result = cv2.erode(img, kernel, iterations=1)
    elif choice == "2":
        result = cv2.dilate(img, kernel, iterations=1)
    elif choice == "3":
        result = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    elif choice == "4":
        result = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    else:
        print("Opción inválida. Inténtalo de nuevo.")
        return morphological_operations(img)

    display_images(img, result, title="Operación Morfológica")

def rellenar_hoyos(imagen, kernel):
    # Convertir la imagen a binaria (Asegúrate de que los agujeros sean 0 y el fondo 255)
    _, imagen_binaria = cv2.threshold(imagen, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Aplicar dilatación para ampliar los elementos y ayudar a llenar los agujeros
    imagen_binaria = cv2.dilate(imagen_binaria, kernel, iterations=1)
    
    # Copiar la imagen binaria para la operación de flood fill
    im_floodfill = imagen_binaria.copy()
    
    # Crear una máscara para el relleno (tamaño de imagen + 2)
    h, w = imagen_binaria.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)
    
    # Aplicar flood fill desde el punto (0, 0) para llenar los agujeros
    cv2.floodFill(im_floodfill, mask, (0, 0), 255)
    
    # Invertir la imagen rellenada para obtener el fondo correcto
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
    
    # Combinar la imagen original con la imagen rellenada
    resultado = imagen_binaria | im_floodfill_inv
    
    return resultado

def boundary_extraction(img):
    """Extract boundaries of objects using the definition: B = A - E(A)."""
    print("Extrayendo límites (Borde del objeto como diferencia entre la imagen original y su erosión)...")
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    eroded = cv2.erode(img, kernel, iterations=1)
    boundary = cv2.subtract(img, eroded)

    display_images(img, boundary, title="Extracción de Límites")

def connected_components(img):
    """Extract connected components using labeling algorithm."""
    print("Extrayendo componentes conectados (Regiones continuas de píxeles en la imagen binaria)...")
    num_labels, labels = cv2.connectedComponents(img)
    print(f"Número de componentes conectados: {num_labels - 1}")

    output = np.zeros_like(img, dtype=np.uint8)
    for label in range(1, num_labels):
        mask = np.uint8(labels == label) * 255
        output = cv2.add(output, mask)

    display_images(img, output, title="Componentes Conectados")

def main():
    """Main menu for user interaction."""
    img = load_image()
    _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    while True:
        print("\nMenú principal:")
        print("1. Operaciones morfológicas básicas")
        print("2. Rellenado de hoyos")
        print("3. Extracción de límites")
        print("4. Extracción de componentes conectados")
        print("5. Salir")
        choice = input("Selecciona una opción: ")

        if choice == "1":
            morphological_operations(binary_img)
        elif choice == "2":
            kernel_size = int(input("Introduce el tamaño del kernel para rellenar hoyos (e.g., 3): "))
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
            result = rellenar_hoyos(binary_img, kernel)
            display_images(binary_img, result, title="Rellenado de Hoyos")
        elif choice == "3":
            boundary_extraction(binary_img)
        elif choice == "4":
            connected_components(binary_img)
        elif choice == "5":
            print("Saliendo del programa...")
            break
        else:
            print("Opción inválida. Inténtalo de nuevo.")

if __name__ == "__main__":
    main()
