import os
from PIL import Image
from tqdm import tqdm

def convert_carpk_to_cnrext(carpk_dir, output_dir, cnrext_label_file):
    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Rutas de las carpetas en CARPK
    annotations_dir = os.path.join(carpk_dir, 'Annotations')
    images_dir = os.path.join(carpk_dir, 'Images')
    imagesets_dir = os.path.join(carpk_dir, 'ImageSets')
    
    # Leer listas de train y test
    with open(os.path.join(imagesets_dir, 'train.txt'), 'r') as f:
        train_images = {line.strip(): 'train' for line in f.readlines()}
    
    with open(os.path.join(imagesets_dir, 'test.txt'), 'r') as f:
        test_images = {line.strip(): 'test' for line in f.readlines()}
    
    # Unificar listas de train y test
    all_images = {**train_images, **test_images}
    
    patch_id = 0  # Contador para los nombres únicos de parches
    
    with open(cnrext_label_file, 'a') as label_file:  # Abrir en modo 'append'
        # Usar tqdm para mostrar el progreso
        for image_name, partition in tqdm(all_images.items(), desc="Procesando imágenes", unit="imagen"):
            # Cargar la imagen correspondiente
            img_path = os.path.join(images_dir, f"{image_name}.png")
            img = Image.open(img_path)
            
            # Leer el archivo de anotación
            annotation_path = os.path.join(annotations_dir, f"{image_name}.txt")
            with open(annotation_path, 'r') as f:
                for line in f.readlines():
                    # Leer coordenadas de la caja delimitadora
                    x_min, y_min, x_max, y_max, _ = map(int, line.split())
                    
                    # Extraer el parche
                    patch = img.crop((x_min, y_min, x_max, y_max))
                    
                    # Guardar el parche con un nombre único
                    patch_filename = f"{partition}_patch_{patch_id}.jpg"
                    patch_path = os.path.join(output_dir, patch_filename)
                    patch.save(patch_path)
                    
                    # Escribir etiqueta (siempre 'ocupado' para CARPK)
                    label_file.write(f"{patch_path} 1\n")
                    patch_id += 1

# Directorio del dataset CARPK
carpk_dir = "Datasets/CARPK_devkit/data"

# Directorio de salida para los parches
output_dir = "Datasets/CARPK_devkit/data/parches_carpk"

# Archivo de etiquetas combinado (de CNR-EXT y PKLot ya convertidos)
cnrext_label_file = "Datasets/CARPK_devkit/data/etiquetas_combined.txt"

# Ejecutar la conversión
convert_carpk_to_cnrext(carpk_dir, output_dir, cnrext_label_file)
