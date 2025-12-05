import os
import cv2
import numpy as np


def verificar_reposo(centros_actuales, centros_previos, umbral_px=4.0):
    
    if len(centros_actuales) != len(centros_previos) or len(centros_actuales) == 0:
        return False

    centros_p_np = np.array(centros_previos)
    
    movimiento_total = 0

    for centro_curr in centros_actuales:
        c_curr = np.array(centro_curr)
        distancias = np.linalg.norm(centros_p_np - c_curr, axis=1)
        distancia_minima = np.min(distancias)
        
        movimiento_total += distancia_minima
    return movimiento_total < umbral_px

def agregar_padding(bbox, padding):
    x, y, w, h = bbox
    
    #Restamos el padding a las coordenadas de origen
    #para mover la esquina superior izquierda
    nuevo_x = x - padding
    nuevo_y = y - padding
    
    #Sumamos el padding*2 para poder compensar la resta anterior
    nuevo_w = w + (padding * 2)
    nuevo_h = h + (padding * 2)
    
    return (nuevo_x, nuevo_y, nuevo_w, nuevo_h)
def segmentar_dados(frame):
    # Convertimos a hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Definimos dos rangos para poder encontrar los dados ya que en el espectro de color
    # los tonos rojos estan al comienzo y al final del espectro.
    # Rango rojo - Tono (0,10), Saturación (150,255), Valor (70,255)
    tono_bajo1 = np.array([0, 150, 70])
    tono_alto1 = np.array([10, 255, 255])

    # Rango Rojo 2 - Tono (170,180) - Saturacion y valor igual al anterior
    tono_bajo2 = np.array([170, 150, 70])
    tono_alto2 = np.array([180, 255, 255])

    # Creamos dos máscaras, una por rango
    mascara_comienzo1 = cv2.inRange(hsv, tono_bajo1, tono_alto1)
    mascara_final2 = cv2.inRange(hsv, tono_bajo2, tono_alto2)
    # Union de máscaras
    mascara = cv2.add(mascara_comienzo1, mascara_final2)

    # Realizamos un cierre para rellenar los dados
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)

    # Una apertura para eliminar ruido
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Encontramos contornos externos
    contours, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    info_dados = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #Separamos contornos que no sean válidos
        if 250 < area < 500:

            x, y, w, h = cv2.boundingRect(cnt)
            cx = x + w // 2
            cy = y + h // 2

            roi = frame[y:y+h, x:x+w]

            dado = {
                "imagen": roi,
                "bounding_box": agregar_padding((x, y, w, h), 5),
                "centro": (cx, cy),
                "area": area
            }
            info_dados.append(dado)

    return frame, mascara, info_dados


def dibujar_dados(frame, info_dados, color, texto_estado):
    # Diagnostico de frames en reposo
    cv2.rectangle(frame, (0, 0), (350, 40), (0, 0, 0), -1)
    cv2.putText(frame, texto_estado, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    for dado in info_dados:
        x, y, w, h = dado["bounding_box"]
        centro = dado["centro"]
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.circle(frame, centro, 3, color, -1)

    return frame


def analizar_tirada(ruta_video):
    cap = cv2.VideoCapture(ruta_video)

    if not cap.isOpened():
        print(f"Error: {ruta_video}")
        return
    #Deben pasar esta cantidad de frames para que se considere en reposo
    frames_para_validar = 5
    centros_previos = []
    contador_reposo = 0
    #Bandera para no llenar la terminal con prints (generaría 1 print por frame)
    ya_leido = False
    pausado = False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    while cap.isOpened():
        if not pausado:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))

            frame_procesado, mascara, info_dados = segmentar_dados(frame)

            centros_actuales = [d["centro"] for d in info_dados]

            centros_actuales.sort()

            esta_quieto = verificar_reposo(
                centros_actuales, centros_previos, umbral_px=10)

            if esta_quieto:
                contador_reposo += 1
            else:
                contador_reposo = 0
                ya_leido = False  # Reset para permitir nueva deteccion

            color_actual = (0, 255, 0)
            texto_estado = f"Cont. Reposo: ({contador_reposo})"

            if contador_reposo > frames_para_validar:
                color_actual = (0, 0, 255)
                texto_estado = f"En reposo - Dados detectados: {len(info_dados)}"

                if not ya_leido:
                    print(f"Dados detectados: {len(info_dados)}")
                    ya_leido = True 

            dibujar_dados(frame_procesado, info_dados,
                          color_actual, texto_estado)

            centros_previos = centros_actuales

            cv2.imshow('Analisis de Dados', frame_procesado)

        key = cv2.waitKey(25) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            pausado = not pausado

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    lista_de_videos = ['tirada_1.mp4', 'tirada_2.mp4',
                       'tirada_3.mp4', 'tirada_4.mp4']

    for video in lista_de_videos:
        analizar_tirada(video)
