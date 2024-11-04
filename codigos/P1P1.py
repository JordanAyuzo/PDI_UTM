#Manipulacion una imagen
## Importacion de bibliotecas
import cv2
import matplotlib.pyplot as plt
import os
def mostrarcv2(nombre, img):
    cv2.imshow(nombre, img)  # muestra la imagen en pantalla
    cv2.waitKey(0)  # espera a que presiones cualquier botón
    cv2.destroyAllWindows()  # opcional: cierra todas las ventanas que tengas abiertas de cv
    
def mostrarPlt(nombre,img):
    plt.imshow(img, cmap='gray')
    plt.title(nombre)
    plt.show()
    
def guardarImagen(nombre,ruta,imagen): 
    if not os.path.exists(ruta):
        os.makedirs(ruta)
    output_path = os.path.join(ruta, nombre) # Ruta de salida
    cv2.imwrite(output_path, imagen)

def main():
    #Cargar una imagen
    ruta = 'codigos/../imagenes/img1.jpg' # Ruta de la imagen
    capa = 0 # Capa de la imagen a cargar (0: escala de grises B(0)G(1)R(2), -1: color)
    img = cv2.imread(ruta,capa)
    
    #mostrar imagen con opencv
    #mostrarcv2('Imagen', img)
    
    #mostrar imagen con matplotlib
    #mostrarPlt('Imagen','mi_ruta',img)
    
    #Guardar imagen
    #guardarImagen('imagen.jpg','mi_ruta',img)
    
    #reducir imagen
    img = cv2.resize(img, (0,0), fx=0.5, fy=0.5)
    mostrarPlt('Imagen Reducida',img)
    img.reshape(-1,3)
    
    
main()