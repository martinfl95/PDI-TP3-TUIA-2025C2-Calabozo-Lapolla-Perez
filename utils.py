import os
import cv2

def visualizar_video(nombre_archivo, factor_redimensionamiento=3):
    """
    Lee un archivo de video, redimensiona los frames y los muestra
    en una ventana. Crea una carpeta 'frames' (si no existe) para
    posibles guardados.

    Args:
        nombre_archivo (str): La ruta o nombre del archivo de video a leer.
        factor_redimensionamiento (int): Factor por el cual dividir el
                                         ancho y alto originales para el
                                         redimensionamiento (por defecto es 3).
    """

    # 1. Preparación del entorno
    os.makedirs("frames", exist_ok=True)
    
    # 2. Leer el video
    cap = cv2.VideoCapture(nombre_archivo)
    
    if not cap.isOpened():
        print(f"ERROR: No se pudo abrir el archivo de video: {nombre_archivo}")
        return

    # Obtener propiedades del video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # fps = int(cap.get(cv2.CAP_PROP_FPS)) # FPS no es necesario para la visualización

    new_width = int(width / factor_redimensionamiento)
    new_height = int(height / factor_redimensionamiento)

    print(f"--- Información del Video ---")
    print(f"Archivo: {nombre_archivo}")
    print(f"Resolución Original: {width}x{height}")
    print(f"Resolución Redimensionada: {new_width}x{new_height}")
    
    # 3. Procesamiento frame a frame
    frame_number = 0
    while (cap.isOpened()):
        ret, frame = cap.read()
        
        if ret == True:
            # Redimensionar el frame
            frame_display = cv2.resize(frame, dsize=(new_width, new_height))
            
            # Mostrar el frame
            cv2.imshow('Frame', frame_display)
            
            # Incrementar contador y esperar tecla
            frame_number += 1
            if cv2.waitKey(25) & 0xFF == ord('q'):
                print(f"Visualización detenida por el usuario en el frame {frame_number}.")
                break
        else: 
            # Se llegó al final del video
            print(f"Fin del video. Frames procesados: {frame_number}.")
            break

    # 4. Limpieza
    cap.release()
    cv2.destroyAllWindows()