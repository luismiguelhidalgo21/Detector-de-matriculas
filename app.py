import cv2
import pytesseract
import threading
import time
import os
import tkinter as tk
from tkinter import ttk, messagebox
from queue import Queue
from PIL import Image, ImageTk
import re

# Configuración de Tesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Ajusta según tu sistema

# Directorio donde se guardarán las imágenes con matrículas detectadas
output_dir = r"C:\Users\prep5\Desktop\project_folder\matriculas_detectadas"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Variables globales
ocr_thread = None
cap = None

# Iniciar la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

# Resto del código...
# Cola para compartir frames entre el hilo principal y el hilo de OCR
frame_queue = Queue(maxsize=10)

# Variable para almacenar el tiempo de la última detección
last_detection_time = time.time() - 60  # Inicializar para permitir una detección inmediata

# Expresión regular para validar matrículas (ajusta según tu formato)
plate_pattern = re.compile(r"^[A-Z]{2,3}\d{3,4}[A-Z]{0,2}$")  # Ejemplo: ABC123 o AB1234CD

# Parámetros para la detección de matrículas
MIN_PLATE_WIDTH = 80  # Ancho mínimo de una matrícula
MIN_PLATE_HEIGHT = 30  # Alto mínimo de una matrícula
MIN_ASPECT_RATIO = 2.0  # Relación de aspecto mínima (ancho/alto)
MAX_ASPECT_RATIO = 6.0  # Relación de aspecto máxima

# Función para preprocesar la imagen
def preprocess_image(image):
    """
    Preprocesa la imagen para mejorar la precisión del OCR.
    :param image: Imagen capturada por la cámara.
    :return: Imagen preprocesada.
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de mediana para reducir el ruido
    gray = cv2.medianBlur(gray, 3)

    # Aplicar suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar umbralización adaptativa
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilatación y erosión para mejorar la calidad de los caracteres
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thresh = cv2.dilate(thresh, kernel, iterations=1)
    thresh = cv2.erode(thresh, kernel, iterations=1)

    return thresh

# Función para detectar regiones de interés (ROI) que podrían ser matrículas
def detect_potential_plates(frame):
    """
    Detecta regiones de interés que podrían ser matrículas.
    :param frame: Frame capturado por la cámara.
    :return: Lista de regiones de interés (ROI).
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Aplicar filtro de mediana para reducir el ruido
    gray = cv2.medianBlur(gray, 3)

    # Detectar bordes usando Canny
    edges = cv2.Canny(gray, 50, 150)

    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos que podrían ser matrículas
    potential_plates = []
    for contour in contours:
        # Obtener el rectángulo del contorno
        x, y, w, h = cv2.boundingRect(contour)

        # Calcular la relación de aspecto
        aspect_ratio = w / h

        # Filtrar por tamaño y relación de aspecto
        if (w >= MIN_PLATE_WIDTH and h >= MIN_PLATE_HEIGHT and
            MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO):
            potential_plates.append((x, y, w, h))

    return potential_plates

# Función para extraer texto con Tesseract (OCR)
def detect_text_from_image(image):
    """
    Extrae texto de una imagen utilizando Tesseract OCR.
    :param image: Imagen preprocesada.
    :return: Texto detectado.
    """
    # Preprocesar la imagen
    processed_image = preprocess_image(image)

    # Configuración de Tesseract
    custom_config = r"--oem 3 --psm 8 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng')

    # Filtrar texto no válido
    text = "".join(char for char in text if char.isalnum()).upper()
    return text.strip()

# Función para validar el formato de la matrícula
def is_valid_plate(text):
    """
    Valida si el texto detectado coincide con el formato de una matrícula.
    :param text: Texto detectado.
    :return: True si es válido, False en caso contrario.
    """
    return bool(plate_pattern.match(text))

# Función para guardar la imagen y el texto detectado
def save_image_with_text(frame, text):
    """
    Guarda la imagen y el texto detectado en el directorio de salida.
    :param frame: Imagen capturada.
    :param text: Texto detectado.
    """
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_image_path = os.path.join(output_dir, f"matricula_{timestamp}.jpg")
    cv2.imwrite(output_image_path, frame)

    with open(os.path.join(output_dir, "matriculas_detectadas.txt"), "a") as f:
        f.write(f"{text}\n")

    print(f"Matrícula detectada y guardada: {text}")

# Función para procesar OCR en un hilo separado
def ocr_worker():
    """
    Procesa los frames de la cola para detectar matrículas utilizando OCR.
    """
    global last_detection_time
    while True:
        try:
            if not frame_queue.empty():
                frame = frame_queue.get()
                if frame is None:
                    break

                current_time = time.time()
                if current_time - last_detection_time >= 60:  # 60 segundos = 1 minuto
                    # Detectar regiones de interés (ROI) que podrían ser matrículas
                    potential_plates = detect_potential_plates(frame)

                    for (x, y, w, h) in potential_plates:
                        # Extraer la región de interés (ROI)
                        roi = frame[y:y+h, x:x+w]

                        # Extraer texto de la imagen con OCR
                        detected_text = detect_text_from_image(roi)

                        # Validar si el texto detectado es una matrícula
                        if detected_text and is_valid_plate(detected_text):
                            print(f"Matrícula detectada: {detected_text}")

                            # Guardar la imagen y el texto detectado
                            save_image_with_text(roi, detected_text)

                            # Actualizar el tiempo de la última detección
                            last_detection_time = current_time
                            print("Un minuto para la siguiente detección")
        except Exception as e:
            print(f"Error en el hilo de OCR: {e}")

# Función para iniciar la grabación
def start_recording():
    """
    Inicia la grabación y el procesamiento de OCR.
    """
    global ocr_thread  # Declara ocr_thread como global
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    record_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    messagebox.showinfo("Grabación", "Grabación iniciada.")
    show_frame()

# Función para detener la grabación
def stop_recording():
    """
    Detiene la grabación y libera los recursos.
    """
    global ocr_thread  # Declara ocr_thread como global
    global cap

    frame_queue.put(None)
    if ocr_thread is not None:
        ocr_thread.join()
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    record_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    messagebox.showinfo("Grabación", "Grabación detenida.")

# Función para ver las imágenes capturadas
def view_images():
    """
    Abre el directorio donde se guardan las imágenes capturadas.
    """
    os.startfile(output_dir)

# Función para actualizar la ventana con el frame de la cámara
def show_frame():
    """
    Muestra el frame de la cámara en la interfaz gráfica.
    """
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        return

    # Detectar regiones de interés (ROI) que podrían ser matrículas
    potential_plates = detect_potential_plates(frame)

    # Dibujar rectángulos alrededor de las regiones detectadas
    for (x, y, w, h) in potential_plates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostrar el frame en la interfaz gráfica
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    # Añadir frame a la cola si no está llena
    if frame_queue.qsize() < 10:
        frame_queue.put(frame)

    # Mostrar el tiempo restante para la siguiente detección
    time_left = max(0, 60 - (time.time() - last_detection_time))
    time_label.config(text=f"Tiempo restante: {int(time_left)} segundos")

    # Reducir la frecuencia de actualización del frame
    label_video.after(30, show_frame)

# Función para manejar el cierre de la aplicación
def on_closing():
    """
    Maneja el cierre de la aplicación liberando los recursos.
    """
    global ocr_thread  # Declara ocr_thread como global
    global cap

    if messagebox.askokcancel("Salir", "¿Estás seguro de que quieres salir?"):
        frame_queue.put(None)
        if ocr_thread is not None and ocr_thread.is_alive():
            ocr_thread.join()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        root.destroy()

# Configuración de la interfaz gráfica (GUI)
root = tk.Tk()
root.title("Detección de Matrículas")
root.geometry("800x600")  # Tamaño de la ventana

# Estilo para los botones
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 12), padding=10)
style.map("TButton",
          foreground=[('pressed', 'white'), ('active', 'white')],
          background=[('pressed', '#4CAF50'), ('active', '#45a049')])

# Etiqueta para mostrar el video
label_video = tk.Label(root)
label_video.pack(pady=20)

# Frame para los botones
button_frame = ttk.Frame(root)
button_frame.pack(pady=10)

# Botones con estilo moderno
record_button = ttk.Button(button_frame, text="Iniciar Grabación", command=start_recording)
record_button.grid(row=0, column=0, padx=10)

stop_button = ttk.Button(button_frame, text="Detener Grabación", command=stop_recording, state=tk.DISABLED)
stop_button.grid(row=0, column=1, padx=10)

view_button = ttk.Button(button_frame, text="Ver Imágenes", command=view_images)
view_button.grid(row=0, column=2, padx=10)

# Añadir una etiqueta para mostrar el tiempo restante
time_label = ttk.Label(root, text="Tiempo restante: 60 segundos", font=("Helvetica", 12))
time_label.pack(pady=10)

# Manejar el cierre de la ventana
root.protocol("WM_DELETE_WINDOW", on_closing)

# Ejecutar la interfaz gráfica
root.mainloop()