import os
import cv2
import numpy as np

def segmentar_dados(frame):
    #Convertimos a hsv
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    #Definimos dos rangos para poder encontrar los dados ya que en el espectro de color
    #los tonos rojos estan al comienzo y al final del espectro.
    #Rango rojo - Tono (0,10), Saturación (150,255), Valor (70,255)
    tono_bajo1 = np.array([0, 150, 70])
    tono_alto1 = np.array([10, 255, 255])
    
    # Rango Rojo 2 - Tono (170,180) - Saturacion y valor igual al anterior
    tono_bajo2 = np.array([170, 150, 70])
    tono_alto2 = np.array([180, 255, 255])

    #Creamos dos máscaras, una por rango
    mascara_comienzo1 = cv2.inRange(hsv, tono_bajo1, tono_alto1)
    mascara_final2 = cv2.inRange(hsv, tono_bajo2, tono_alto2)
    mascara = cv2.add(mascara_comienzo1, mascara_final2)

    #Realizamos un cierre para rellenar los dados
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE,kernel) 
    
    #Una apertura para eliminar ruido
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)
    
    #Encontramos contornos externos
    contours, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    dados_detectados = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        #Los dados tienen un area determinada asique realizamos un filtro
        #Si filtramos area con valores más chicos que 700 hace que deje de trackear dados consistentemente
        if area > 325 and area < 700:
            #Encontramos el bounding box
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Dibujar rectángulo en el frame original
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Dado", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            roi = frame[y:y+h, x:x+w]
            dados_detectados.append(roi)
    return frame, mascara, dados_detectados

if __name__ == '__main__':
    # os.makedirs("frames", exist_ok = True)
    cap = cv2.VideoCapture('tirada_1.mp4')
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_inicio = int(fps * 1) 
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    pausado = False
    while (cap.isOpened()):
        if not pausado:
            ret, frame = cap.read()
            #Loop, apretar Q para salir
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_inicio)
                continue
            frame = cv2.resize(frame, dsize=(int(width/3), int(height/3)))
            frame_procesado, mascara_binaria, lista_dados = segmentar_dados(frame)

        else:
            frame_pausa = frame.copy()
            #Frame de pausa
            cv2.imshow('Video', frame_pausa)
        
        if not pausado:
            #Reproducción de video normal
            cv2.imshow('Deteccion', frame_procesado)
            cv2.imshow('Mascara', mascara_binaria)

        #Delay 0 = pausa
        delay = 0 if pausado else 25
        key = cv2.waitKey(delay) & 0xFF

        #Cierre (Apretar 'q')
        if key == ord('q'):
            break
        #Pausa (Apretar 'p')
        elif key == ord('p'):
            pausado = not pausado
    cap.release()
    cv2.destroyAllWindows()