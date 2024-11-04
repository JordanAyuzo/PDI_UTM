import numpy as np
import cv2

def mostrar(nombre, img):
    cv2.imshow(nombre, img)  # muestra la imagen en pantalla
    cv2.waitKey(0)  # espera a que presiones cualquier botón
    cv2.destroyAllWindows()  # opcional: cierra todas las ventanas que tengas abiertas de cv2

def redimensionar(img, tam):
    return cv2.resize(img, (tam, tam))  # Cambia para que el tamaño sea cuadrado

def comoes(img):
    print("Tamaño:" , img.shape)  # devuelve pixel,pixel,canales
    print("  max :" , np.max(img))  # el número máximo 
    print("  min :" , np.min(img))  # devuelve el mínimo de la matriz

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

def main():
    circulo = cv2.imread('codigos/../imagenes/circulo.jpg', 0)  # Carga la imagen que necesitas
    cuadrado = cv2.imread('codigos/../imagenes/cuadrado.jpg', 0)
    triangulo = cv2.imread('codigos/../imagenes/triangulo.png', 0)

    # Redimensionar a 300x300 píxeles
    circulo = redimensionar(circulo, 300)
    cuadrado = redimensionar(cuadrado, 300)
    triangulo = redimensionar(triangulo, 300)

    print('Imagen Original(ENTRADA):')
    mostrar('Circulo', circulo)
    mostrar('Cuadrado', cuadrado)
    mostrar('Triangulo', triangulo)

    not_circulo = operador_not(circulo)
    mostrar('NOT Circulo', not_circulo)

    and_img = operador_and(circulo, cuadrado)
    mostrar('AND Circulo-Cuadrado', and_img)

    or_img = operador_or(circulo, triangulo)
    mostrar('OR Circulo-Triangulo', or_img)

    nand_img = operador_nand(circulo, cuadrado)
    mostrar('NAND Circulo-Cuadrado', nand_img)

    xor_img = operador_xor(cuadrado, triangulo)
    mostrar('XOR Cuadrado-Triangulo', xor_img)

main()
