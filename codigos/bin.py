import cv2

def convertir_a_binaria(ruta_imagen, umbral=170):
    """
    Convierte una imagen normal en una imagen binaria.
    
    :param ruta_imagen: Ruta de la imagen de entrada.
    :param umbral: Valor de umbral para binarización (0-255). Por defecto, 127.
    """
    # Cargar la imagen
    imagen = cv2.imread(ruta_imagen)
    
    if imagen is None:
        print("No se pudo cargar la imagen. Verifica la ruta.")
        return
    
    # Convertir a escala de grises
    imagen_gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el umbral para binarizar
    _, imagen_binaria = cv2.threshold(imagen_gris, umbral, 255, cv2.THRESH_BINARY)
    
    # Mostrar las imágenes
    cv2.imshow("Imagen Original", imagen)
    cv2.imshow("Imagen Binaria", imagen_binaria)
    
    # Guardar la imagen binaria
    salida_binaria = "imagen_binaria.png"
    cv2.imwrite(salida_binaria, imagen_binaria)
    print(f"Imagen binaria guardada como {salida_binaria}")
    
    # Esperar una tecla y cerrar las ventanas
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Ejemplo de uso
ruta = "codigos/../imagenes/img.jpg"  # Cambia esto por la ruta de tu imagen
convertir_a_binaria(ruta)
