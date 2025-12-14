import os
import cv2
import numpy as np
import copy

frame_actual_click = None
frame_actual_index = 0


def verificar_reposo(centros_actuales, centros_previos, umbral_px=4.0):
    if len(centros_actuales) != len(centros_previos) or len(centros_actuales) == 0:
        return False

    centros_prev = np.array(centros_previos)
    movimiento_total = 0

    for centro_curr in centros_actuales:
        c_curr = np.array(centro_curr)
        distancias = np.linalg.norm(centros_prev - c_curr, axis=1)
        distancia_minima = np.min(distancias)
        movimiento_total += distancia_minima

    return movimiento_total < umbral_px


def contar_puntos(region_interes, debug, mostrar_detalle):
    alto_reg, ancho_reg = region_interes.shape[:2]
    area_total_reg = alto_reg * ancho_reg

    img_debug = region_interes.copy()

    img_hsv = cv2.cvtColor(region_interes, cv2.COLOR_BGR2HSV)
    # Filtrado por rango de color blanco
    bajo_blanco = np.array([0, 0, 95])
    alto_blanco = np.array([180, 80, 255])

    mascara = cv2.inRange(img_hsv, bajo_blanco, alto_blanco)

    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    puntos = 0
    for contorno in contornos:
        area = cv2.contourArea(contorno)

        # Filtro de ruido
        if area < 3:
            continue

        if (area_total_reg * 0.001) < area < (area_total_reg * 0.2):
            contorno = cv2.convexHull(contorno)
            area_contorno = cv2.contourArea(contorno)
            perimetro_contorno = cv2.arcLength(contorno, True)

            if perimetro_contorno == 0:
                continue

            factor_forma = (perimetro_contorno *
                            perimetro_contorno) / (4 * np.pi * area_contorno)

            # Validación por factor de forma laxo - no obtenemos contornos regulares
            if factor_forma < 2.1:
                # Contorno aceptado - color verde
                puntos += 1
                cv2.drawContours(img_debug, [contorno], -1, (0, 255, 0), 1)
            else:
                # Falla por forma - color rojo
                cv2.drawContours(img_debug, [contorno], -1, (0, 0, 255), 1)
        else:
            # Falla por tamaño - color azul
            cv2.drawContours(img_debug, [contorno], -1, (255, 0, 0), 1)

    # Flag que permite la visualización de los contornos encontrados sobre el dado
    if debug and mostrar_detalle:
        alto, ancho = img_debug.shape[:2]
        # Reescalada porque si no, queda muy chica para poder apreciar el detalle
        img_grande = cv2.resize(
            img_debug, (ancho * 10, alto * 10), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Debug - Presiona tecla para continuar", img_grande)
        # Esperamos que se presione una tecla para poder continuar, necesitamos esto para visualizar
        # cada dado y no solamente el último.
        cv2.waitKey(0)

    # Si tenemos más de 6 puntos estamos en la lona
    return puntos if 0 < puntos <= 6 else 0


def agregar_margen(bounding_box, margen):
    x, y, w, h = bounding_box
    # Restamos el padding a las coordenadas de origen
    # para mover la esquina superior izquierda
    nuevo_x = x - margen
    nuevo_y = y - margen
    # Sumamos el padding*2 para poder compensar la resta anterior
    nuevo_w = w + (margen * 2)
    nuevo_h = h + (margen * 2)
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
    nucleo = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (13, 13))
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, nucleo)

    # Una apertura para eliminar ruido
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, nucleo)
    cv2.imshow("Mascara Dados", mascara)

    # Encontramos contornos externos
    contornos, _ = cv2.findContours(
        mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    info_dados = []

    for contorno in contornos:
        area = cv2.contourArea(contorno)
        # Separamos contornos que no sean válidos
        if 250 < area < 500:
            x, y, w, h = cv2.boundingRect(contorno)
            centro_x = x + w // 2
            centro_y = y + h // 2

            region_interes = frame[y:y+h, x:x+w]

            dado = {
                "imagen": region_interes,
                "bounding_box": agregar_margen((x, y, w, h), 5),
                "centro": (centro_x, centro_y),
                "area": area,
                "puntos": 0,
            }
            info_dados.append(dado)

    return frame, mascara, info_dados


def dibujar_dados(frame, info_dados, color, texto_estado, mostrar_valor=False, suma_total=0, factor_escala=1):
    # Barra de estado superior
    cv2.rectangle(frame, (0, 0), (int(600 * factor_escala),
                  int(40 * factor_escala)), (0, 0, 0), -1)
    cv2.putText(frame, texto_estado, (int(10 * factor_escala), int(25 * factor_escala)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7 * factor_escala, color, 2 * factor_escala)

    for i, dado in enumerate(info_dados):
        x, y, w, h = dado["bounding_box"]
        centro_x, centro_y = dado["centro"]
        id_dado = i + 1

        # Bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2 * factor_escala)

        # ID - ¿arriba de la bounding box?
        cv2.putText(frame, f"ID:{id_dado}", (x, y - int(5 * factor_escala)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5 * factor_escala, color, 1 * factor_escala)

        # Lógica para mostrar el valor en el centro de la bounding box
        if mostrar_valor:
            texto_valor = str(dado['puntos'])
            # Configuración de fuente
            escala_fuente = 1.0 * factor_escala
            grosor = 2 * factor_escala
            fuente = cv2.FONT_HERSHEY_SIMPLEX

            # Obtenemos el tamaño del texto (valor del dado)
            (ancho_texto, alto_texto), linea_base = cv2.getTextSize(
                texto_valor, fuente, escala_fuente, grosor)

            # Calculamos la posición exacta
            # Restamos la mitad del ancho del texto al centro X
            texto_x = int(centro_x - (ancho_texto / 2))
            # Sumamos la mitad del alto del texto al centro Y
            texto_y = int(centro_y + (alto_texto / 2))

            cv2.putText(frame, texto_valor, (texto_x, texto_y),
                        fuente, escala_fuente, (255, 255, 0), grosor)
        else:
            # Si no está en reposo, dibujamos el centro del dado.
            cv2.circle(frame, (centro_x, centro_y),
                       3 * factor_escala, color, -1)
    return frame


def onClick(evento, x, y, flags, param):
    '''
    Evento que captura el click izquierdo del mouse dentro de la ventana 'Frames Originales'
    Retorna por consola:
    - Número de Frame del video
    - Coordenadas del click
    - Valores BGR : (Blue, Green, Red)
    - Valores HSV : (Tono, Saturación, Valor)
    '''
    global frame_actual_click
    if evento == cv2.EVENT_LBUTTONDOWN:
        if frame_actual_click is not None:
            # Validar que las coordenadas estén dentro de la imagen
            alto, ancho = frame_actual_click.shape[:2]
            if x < ancho and y < alto:
                # Obtener valor BGR
                bgr_pixel = frame_actual_click[y, x]
                # Convertir pixel a HSV
                pixel_array = np.uint8([[bgr_pixel]])
                hsv_pixel = cv2.cvtColor(pixel_array, cv2.COLOR_BGR2HSV)[0][0]
                h, s, v = hsv_pixel
                print(f"FRAME N°: {frame_actual_index}")
                print(f"Click en (x={x}, y={y})")
                print(f"BGR: {bgr_pixel}")
                print(f"HSV: [H: {h}, S: {s}, V: {v}]")
                print("-" * 30)

def analizar_tirada(ruta_video, grabar_datos, mostrar_detalle_puntos):
    captura = cv2.VideoCapture(ruta_video)
    nombre_archivo = os.path.basename(ruta_video)
    nombre_sin_ext = os.path.splitext(nombre_archivo)[0]
    print(f'--- Video: {ruta_video} ---')
    if not captura.isOpened():
        print(f"Error: {ruta_video}")
        return

    # Variable global para evento onClick
    global frame_actual_click
    # Variable global para llevar el conteo de frames
    global frame_actual_index
    cv2.namedWindow('Frames originales')
    cv2.setMouseCallback('Frames originales', onClick)

    # Deben pasar esta cantidad de frames para que se considere en reposo
    frames_para_validar = 5
    centros_previos = []
    contador_reposo = 0
    # Bandera para no llenar la terminal con prints (generaría 1 print por frame)
    ya_leido = False
    # Bandera para pausa de video
    pausado = False
    # Bandera para observar cada dado detectado
    debug_dado = False
    valores_detectados = {}

    # Datos del video para resizing de procesamiento y grabado
    ancho = int(captura.get(cv2.CAP_PROP_FRAME_WIDTH))
    alto = int(captura.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = captura.get(cv2.CAP_PROP_FPS)

    grabador_video = None
    # Creación de rutas
    dir_frames_orig = os.path.join("frames_originales", nombre_sin_ext)
    dir_frames_proc = os.path.join("frames_procesados", nombre_sin_ext)
    dir_videos_proc = "videos_procesados"

    # Factor de escala inverso (ya que redimensionamos a 1/3)
    factor_escala = 3

    # Creación de carpetas
    if grabar_datos:
        os.makedirs(dir_frames_orig, exist_ok=True)
        os.makedirs(dir_frames_proc, exist_ok=True)
        os.makedirs(dir_videos_proc, exist_ok=True)
        ruta_video_salida = os.path.join(
            dir_videos_proc, f"tirada_procesada_{nombre_sin_ext.split('_')[-1]}.mp4")
        #Instancia de VideoWriter para guardar el video procesado
        codigo_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        grabador_video = cv2.VideoWriter(
            ruta_video_salida, codigo_fourcc, fps, (ancho, alto))

    indice_frame = 0

    while captura.isOpened():
        if not pausado:
            exito, frame_completo = captura.read()
            if not exito:
                break
            
            frame_escalado = cv2.resize(frame_completo, dsize=(
                int(ancho/factor_escala), int(alto/factor_escala)))

            # Update del frame e indice para poder utilizar evento onClick
            frame_actual_click = frame_escalado
            frame_actual_index = indice_frame

            # Guardado de frame original
            if grabar_datos:
                cv2.imwrite(os.path.join(dir_frames_orig,
                            f"frame_{indice_frame}.jpg"), frame_completo)

            cv2.imshow('Frames originales', frame_escalado)

            # Procesamos sobre la imagen escalada
            _, mascara, info_dados = segmentar_dados(frame_escalado)

            #Obtenemos los centros de los dados
            info_dados.sort(key=lambda d: d["centro"][0])
            centros_actuales = [d["centro"] for d in info_dados]
            centros_para_calculo = sorted(centros_actuales)

            esta_quieto = verificar_reposo(
                centros_para_calculo, centros_previos, umbral_px=10)

            #Contador > 5 (frames_para validar) corresponde a reposo
            if esta_quieto:
                contador_reposo += 1
            else:
                contador_reposo = 0
                ya_leido = False
                valores_detectados = {}

            #Estado a modo de barra superior en el frame
            color_actual = (0, 255, 0)
            texto_estado = f"Cont. Reposo: ({contador_reposo})"
            estable = False
            suma_total = 0

            if contador_reposo > frames_para_validar:
                #Solo nos interesa hacer el análisis si detectamos dados
                if len(info_dados) > 0:
                    color_actual = (0, 0, 255)
                    texto_estado = f"En reposo - Dados: {len(info_dados)}"
                    estable = True
                    for i, dado in enumerate(info_dados):
                        if i in valores_detectados:
                            dado['puntos'] = valores_detectados[i]
                        else:
                            puntos = contar_puntos(
                                dado["imagen"], debug_dado, mostrar_detalle_puntos)
                            dado['puntos'] = puntos
                            valores_detectados[i] = puntos

                        suma_total += dado['puntos']

                    if not ya_leido:
                        print(
                            f"--- Reposo alcanzado en Frame: {indice_frame} ---")
                        for i, dado in enumerate(info_dados):
                            print(f"Valor del dado {i+1}: {dado['puntos']}")

                        print(
                            f"Dados detectados: {len(info_dados)} | Suma Total: {suma_total}")
                        print("-" * 30)
                        ya_leido = True

            dibujar_dados(frame_escalado, info_dados, color_actual,
                          texto_estado, estable, suma_total)
            cv2.imshow('Analisis de Dados', frame_escalado)

            if grabar_datos:
                # Creamos copia profunda para no afectar los datos del frame escalado
                info_dados_completo = copy.deepcopy(info_dados)
                # Escalamos coordenadas para graficar en la resolución original
                for d in info_dados_completo:
                    x, y, w, h = d["bounding_box"]
                    cx, cy = d["centro"]
                    d["bounding_box"] = (
                        x * factor_escala, y * factor_escala, w * factor_escala, h * factor_escala)
                    d["centro"] = (cx * factor_escala, cy * factor_escala)

                # Dibujamos sobre el frame original
                frame_completo_proc = frame_completo.copy()
                dibujar_dados(frame_completo_proc, info_dados_completo,
                              color_actual, texto_estado, estable, suma_total, factor_escala)

                # Guardamos frame procesado y guardamos video
                cv2.imwrite(os.path.join(dir_frames_proc,
                            f"frame_{indice_frame}.jpg"), frame_completo_proc)
                grabador_video.write(frame_completo_proc)

            centros_previos = centros_para_calculo
            indice_frame += 1
        #Eventos:
        # 'q' cortar ejecucion video
        # 'p' pausar video
        tecla = cv2.waitKey(25) & 0xFF
        if tecla == ord('q'):
            break
        elif tecla == ord('p'):
            pausado = not pausado

    captura.release()
    if grabador_video:
        grabador_video.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # True: Guardar los frames originales y los frames/videos procesados
    # False: Observar el funcionamiento del algoritmo sin persistir datos
    GUARDAR_DATOS = True

    # True: Visualizar la detección del valor de cada dado - presionar cualquier tecla para continuar
    # False: Visualizar video procesado sin pausas
    MOSTRAR_DETALLE_PUNTOS = False

    lista_videos = ['tirada_1.mp4', 'tirada_2.mp4',
                    'tirada_3.mp4', 'tirada_4.mp4']

    # Funcionalidades extra:
    # Presionar 'p' para pausar video
    # Presionar 'q' para terminar la ejecución del video
    # Evento 'onClick' - clickear sobre cualquier lugar del display de frames originales
    # devuelve el numero de frame, coordenada de click y el tono, saturación y valor del pixel
    for video in lista_videos:
        analizar_tirada(video, GUARDAR_DATOS, MOSTRAR_DETALLE_PUNTOS)
