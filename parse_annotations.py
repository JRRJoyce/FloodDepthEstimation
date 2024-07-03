import os
import numpy as np

# Load your YOLO annotations, typically they are in .txt files with the same base name as your images
def load_annotations(annotation_file):
    with open(annotation_file, 'r') as f:
        annotations = f.readlines()
    return annotations

def parse_annotations(annotations):
    parsed_annotations = []
    for annotation in annotations:
        parts = annotation.strip().split()
        class_id = int(parts[0])
        # Extract pairs of coordinates from the remaining parts
        polygon = [(float(parts[i]), float(parts[i+1])) for i in range(1, len(parts), 2)]
        parsed_annotations.append((class_id, polygon))
    return parsed_annotations

"""
def parse_yolo_annotations(annotation, img_width, img_height):
    coords = list(map(float, annotation.split()[1:]))  # Skip the first element as it's the class label (0 for water)
    points = []
    for i in range(0, len(coords), 2):
        x = int(coords[i] * img_width)
        y = int(coords[i + 1] * img_height)
        points.append((x, y))
    return np.array(points, dtype=np.int32)
"""

def filter_images_with_classes(image_dir, annotation_dir, water_class_id, traffic_sign_class_id):
    selected_images = []

    for annotation_file in os.listdir(annotation_dir):
        if annotation_file.endswith('.txt'):
            annotations = load_annotations(os.path.join(annotation_dir, annotation_file))
            parsed_annotations = parse_annotations(annotations)

            has_water = any(ann[0] == water_class_id for ann in parsed_annotations)
            has_traffic_sign = any(ann[0] == traffic_sign_class_id for ann in parsed_annotations)

            if has_water and has_traffic_sign:
                image_base_name = os.path.splitext(annotation_file)[0]
                image_file = os.path.join(image_dir, f"{image_base_name}.jpg")
                if os.path.exists(image_file):
                    selected_images.append(image_file)

    return selected_images