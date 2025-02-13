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

# Iniciar la cámara
cap = cv2.VideoCapture(0)  # 0 para la cámara predeterminada

if not cap.isOpened():
    print("Error: No se pudo acceder a la cámara.")
    exit()

# Cola para compartir frames entre el hilo principal y el hilo de OCR
frame_queue = Queue(maxsize=10)

# Variable para almacenar el tiempo de la última detección
last_detection_time = time.time() - 60  # Inicializar para permitir una detección inmediata

# Expresión regular para validar matrículas (ajusta según tu formato)
plate_pattern = re.compile(r"^[A-Z0-9]{6,10}$")  # Ejemplo: ABC123 o 1234XYZ

# Función para preprocesar la imagen
def preprocess_image(image):
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Aplicar suavizado para reducir el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar umbralización adaptativa
    thresh = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    return thresh

# Función para extraer texto con Tesseract (OCR)
def detect_text_from_image(image):
    # Preprocesar la imagen
    processed_image = preprocess_image(image)

    # Configuración de Tesseract
    custom_config = r"--oem 3 --psm 8"  # PSM 8: Trata la imagen como una sola palabra
    text = pytesseract.image_to_string(processed_image, config=custom_config, lang='eng')

    # Filtrar texto no válido
    text = "".join(char for char in text if char.isalnum()).upper()
    return text.strip()

# Función para validar el formato de la matrícula
def is_valid_plate(text):
    return bool(plate_pattern.match(text))

# Función para guardar la imagen y el texto detectado
def save_image_with_text(frame, text):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_image_path = os.path.join(output_dir, f"matricula_{timestamp}.jpg")
    cv2.imwrite(output_image_path, frame)

    with open(os.path.join(output_dir, "matriculas_detectadas.txt"), "a") as f:
        f.write(f"{text}\n")

    print(f"Matrícula detectada y guardada: {text}")

# Función para procesar OCR en un hilo separado
def ocr_worker():
    global last_detection_time
    while True:
        if not frame_queue.empty():
            frame = frame_queue.get()
            if frame is None:
                break

            current_time = time.time()
            if current_time - last_detection_time >= 60:  # 60 segundos = 1 minuto
                # Extraer texto de la imagen con OCR
                detected_text = detect_text_from_image(frame)

                # Validar si el texto detectado es una matrícula
                if detected_text and is_valid_plate(detected_text):
                    print(f"Matrícula detectada: {detected_text}")

                    # Guardar la imagen y el texto detectado
                    save_image_with_text(frame, detected_text)

                    # Actualizar el tiempo de la última detección
                    last_detection_time = current_time
                    print("Un minuto para la siguiente detección")

# Función para iniciar la grabación
def start_recording():
    global ocr_thread
    ocr_thread = threading.Thread(target=ocr_worker, daemon=True)
    ocr_thread.start()

    record_button.config(state=tk.DISABLED)
    stop_button.config(state=tk.NORMAL)
    messagebox.showinfo("Grabación", "Grabación iniciada.")
    show_frame()

# Función para detener la grabación
def stop_recording():
    global cap
    frame_queue.put(None)
    ocr_thread.join()
    cap.release()
    cv2.destroyAllWindows()
    record_button.config(state=tk.NORMAL)
    stop_button.config(state=tk.DISABLED)
    messagebox.showinfo("Grabación", "Grabación detenida.")

# Función para ver las imágenes capturadas
def view_images():
    os.startfile(output_dir)

# Función para actualizar la ventana con el frame de la cámara
def show_frame():
    ret, frame = cap.read()
    if not ret:
        print("Error: No se pudo capturar el frame.")
        return

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)

    label_video.imgtk = imgtk
    label_video.configure(image=imgtk)

    if frame_queue.qsize() < 10:
        frame_queue.put(frame)

    label_video.after(10, show_frame)

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

# Ejecutar la interfaz gráfica
root.mainloop()