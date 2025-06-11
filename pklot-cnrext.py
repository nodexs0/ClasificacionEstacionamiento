import json
import os
from PIL import Image
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def process_annotation(ann, images_dict, output_dir, partition, patch_id_start):
    """Procesa una anotación individual y extrae un parche."""
    image_id = ann['image_id']
    category_id = ann['category_id']
    
    # Solo categorías que nos interesan
    if category_id == 1:
        label = 0
    elif category_id == 2:
        label = 1
    else:
        return None  # Ignorar anotaciones no relevantes
    
    img_path = images_dict[image_id]
    img = Image.open(img_path)
    
    # bbox: [x, y, width, height]
    x, y, w, h = map(int, ann['bbox'])
    
    # Extraer parche
    patch = img.crop((x, y, x + w, y + h))
    
    # Guardar parche con nombre único
    patch_filename = f"{partition}_patch_{patch_id_start}.jpg"
    patch_path = os.path.join(output_dir, patch_filename)
    patch.save(patch_path)
    
    return patch_filename, label

def convert_and_extract_patches_parallel(pklot_dir, output_dir, output_file):
    partitions = ['train', 'test', 'val']
    patch_id = 0
    
    with open(output_file, 'w') as f_out:
        for partition in partitions:
            annotation_file = os.path.join(pklot_dir, partition, '_annotations.coco.json')
            with open(annotation_file, 'r') as f:
                data = json.load(f)
            
            # Mapear imagen id a su ruta completa
            images_dict = {img['id']: os.path.join(pklot_dir, partition, img['file_name']) for img in data['images']}
            
            # Usar un pool de procesos para paralelizar el trabajo
            results = []
            with ProcessPoolExecutor() as executor:
                # Enumerar anotaciones para asignar IDs únicos
                tasks = [
                    executor.submit(process_annotation, ann, images_dict, output_dir, partition, patch_id + i)
                    for i, ann in enumerate(data['annotations'])
                ]
                for future in tqdm(tasks, desc=f"Procesando {partition}"):
                    result = future.result()
                    if result is not None:
                        results.append(result)
            
            # Escribir resultados en el archivo de etiquetas
            for filename, label in results:
                f_out.write(f"{filename} {label}\n")
            
            patch_id += len(results)

if __name__ == '__main__':
    pklot_dir = "Datasets/PKLot"
    output_dir = "Datasets/PKLot/parches_extraidos"
    os.makedirs(output_dir, exist_ok=True)
    output_file = "Datasets/PKLot/pklot_patches_labels.txt"

    convert_and_extract_patches_parallel(pklot_dir, output_dir, output_file)