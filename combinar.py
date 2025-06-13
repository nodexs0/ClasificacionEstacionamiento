import os
import shutil
import random
from multiprocessing import Process, Manager

def process_dataset(dataset_index, patch_dir, annotation_file, output_dir,
                    train_ratio, val_ratio, test_ratio, return_counts):

    # Crear carpetas de salida (deben existir antes de paralelizar, o usar lock)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    test_dir = os.path.join(output_dir, 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    dataset = []
    with open(annotation_file, 'r') as f:
        for line in f.readlines():
            image_path, label = line.strip().split()
            # Construir ruta absoluta a imagen
            full_img_path = os.path.join(patch_dir, image_path)
            dataset.append((full_img_path, label))

    random.shuffle(dataset)

    total = len(dataset)
    train_cutoff = int(total * train_ratio)
    val_cutoff = int(total * (train_ratio + val_ratio))

    train_data = dataset[:train_cutoff]
    val_data = dataset[train_cutoff:val_cutoff]
    test_data = dataset[val_cutoff:]

    counts = {'train': 0, 'val': 0, 'test': 0}

    # Función para copiar y escribir etiquetas sin barra de progreso,
    # con mensajes al inicio y fin
    def copy_and_write(data, folder, label_path, part_name):
        print(f"[Dataset {dataset_index}] Iniciando copia de {len(data)} imágenes para {part_name}")
        with open(label_path, 'w') as label_file:
            for img_path, label in data:
                base_name = os.path.basename(img_path)
                unique_name = f"dataset{dataset_index}_{base_name}"
                dst_path = os.path.join(folder, unique_name)
                shutil.copy(img_path, dst_path)
                label_file.write(f"{dst_path} {label}\n")
        print(f"[Dataset {dataset_index}] Terminó copia para {part_name} ({len(data)} imágenes)")
        counts[part_name] = len(data)

    # Archivos temporales de etiquetas por dataset y partición
    train_label_path = os.path.join(output_dir, f"train_labels_dataset{dataset_index}.txt")
    val_label_path = os.path.join(output_dir, f"val_labels_dataset{dataset_index}.txt")
    test_label_path = os.path.join(output_dir, f"test_labels_dataset{dataset_index}.txt")

    copy_and_write(train_data, train_dir, train_label_path, 'train')
    copy_and_write(val_data, val_dir, val_label_path, 'val')
    copy_and_write(test_data, test_dir, test_label_path, 'test')

    return_counts[dataset_index] = counts


def combine_label_files(output_dir, num_datasets):
    # Archivos finales
    final_train = os.path.join(output_dir, 'train_labels.txt')
    final_val = os.path.join(output_dir, 'val_labels.txt')
    final_test = os.path.join(output_dir, 'test_labels.txt')

    with open(final_train, 'w') as ft, open(final_val, 'w') as fv, open(final_test, 'w') as fte:
        for i in range(num_datasets):
            train_tmp = os.path.join(output_dir, f"train_labels_dataset{i}.txt")
            val_tmp = os.path.join(output_dir, f"val_labels_dataset{i}.txt")
            test_tmp = os.path.join(output_dir, f"test_labels_dataset{i}.txt")

            for ftmp, ffinal in [(train_tmp, ft), (val_tmp, fv), (test_tmp, fte)]:
                with open(ftmp, 'r') as fr:
                    shutil.copyfileobj(fr, ffinal)
                os.remove(ftmp)  # eliminar archivo temporal


if __name__ == "__main__":
    patch_dirs = [
        'Datasets/CNR-EXT-Patches-150x150/PATCHES',
        'Datasets/PKLot/parches_extraidos',
        ''
    ]

    annotation_files = [
        'Datasets/CNR-EXT-Patches-150x150/LABELS/all.txt',
        'Datasets/PKLot/pklot_patches_labels.txt',
        'Datasets/CARPK_devkit/data/etiquetas_combined.txt'
    ]

    output_dir = 'Datasets/dataset_combinado'
    os.makedirs(output_dir, exist_ok=True)

    train_ratio = 0.7
    val_ratio = 0.2
    test_ratio = 0.1

    manager = Manager()
    return_counts = manager.dict()

    processes = []
    for i, (patch_dir, ann_file) in enumerate(zip(patch_dirs, annotation_files)):
        p = Process(target=process_dataset,
                    args=(i, patch_dir, ann_file, output_dir,
                          train_ratio, val_ratio, test_ratio, return_counts))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    combine_label_files(output_dir, len(patch_dirs))

    print("Conjunto combinado creado.")
    for i in range(len(patch_dirs)):
        print(f"Dataset {i} particiones:", return_counts[i])
