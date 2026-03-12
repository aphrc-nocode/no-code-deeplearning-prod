
import os
import yaml
import shutil
from tqdm import tqdm

def get_yolo_cache_dir(data_dir):
    """Returns the path to the YOLO cache directory within the dataset directory."""
    return os.path.join(data_dir, "yolo_cache")

def is_yolo_cache_valid(cache_dir, split):
    """Simple check to see if the cache directory for a split exists and is not empty."""
    split_dir = os.path.join(cache_dir, split)
    if not os.path.exists(split_dir):
        return False
    
    # Check for images and labels folders
    images_dir = os.path.join(split_dir, "images")
    labels_dir = os.path.join(split_dir, "labels")
    
    if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
        return False
        
    # Could add more sophisticated checks (e.g. file count match), but this is a good start
    if not os.listdir(images_dir):
        return False
        
    return True

def convert_split_to_yolo(dataset, output_path, split):
    """
    Converts a Hugging Face dataset split into the YOLOv5 TXT format.
    Creates 'images' and 'labels' folders.
    """
    image_dir = os.path.join(output_path, split, "images")
    label_dir = os.path.join(output_path, split, "labels")
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    print(f"Converting {split} split to YOLO format at {output_path}...")
    
    # Get class names from features
    try:
        class_names = dataset.features['objects']['category'].feature.names
    except:
        # Fallback or handle different structures if necessary
        class_names = [] # This strictly expects our standard format

    for i in tqdm(range(len(dataset)), desc=f"Converting {split}"):
        example = dataset[i]
        image = example["image"]
        
        # Use image_id if available, otherwise fall back to index
        image_id_str = str(example.get("image_id", i))
        
        # Save the image
        image_filename = f"{image_id_str}.jpg"
        image.save(os.path.join(image_dir, image_filename))

        # Create the label file
        label_filename = f"{image_id_str}.txt"
        with open(os.path.join(label_dir, label_filename), "w") as f:
            objects = example["objects"]
            img_width = example["width"]
            img_height = example["height"]

            for category, bbox in zip(objects["category"], objects["bbox"]):
                # Bbox is [x_min, y_min, w, h] (COCO format)
                x_min, y_min, w, h = bbox
                
                # Convert to YOLO format (center_x, center_y, width, height) normalized
                x_center_norm = (x_min + w / 2) / img_width
                y_center_norm = (y_min + h / 2) / img_height
                w_norm = w / img_width
                h_norm = h / img_height
                
                f.write(f"{category} {x_center_norm} {y_center_norm} {w_norm} {h_norm}\n")
    
    return class_names

def prepare_yolo_data(dataset, data_dir, output_dir):
    """
    Orchestrates the preparation of YOLO data.
    1. Checks for cached YOLO data in data_dir/yolo_cache.
    2. If missing, converts and caches it.
    3. Symlinks or copies the data to the current run's output_dir (or just uses it directly).
    
    Returns the path to the data.yaml file.
    """
    cache_dir = get_yolo_cache_dir(data_dir)
    os.makedirs(cache_dir, exist_ok=True)
    
    splits = ["train", "validation"]
    if "test" in dataset:
        splits.append("test")
        
    try:
        class_names = dataset["train"].features['objects']['category'].feature.names
    except Exception:
        # Fallback if the dataset was instantiated without explicit ClassLabels
        max_class_id = 0
        for item in dataset["train"]:
            cats = item.get("objects", {}).get("category", [])
            if len(cats) > 0:
                max_class_id = max(max_class_id, max(cats))
        
        class_names = [f"class_{i}" for i in range(max_class_id + 1)]
        print(f"Warning: Could not extract ClassLabels explicitly from features. Generating generic classes up to ID {max_class_id}")

    # Check and Create Cache if needed
    for split in splits:
        if is_yolo_cache_valid(cache_dir, split):
            print(f"Found valid YOLO cache for {split} in {cache_dir}. Using cache.")
        else:
            print(f"No valid cache for {split}. Converting...")
            # We convert directly into the cache directory
            convert_split_to_yolo(dataset[split], cache_dir, split)

    # Now create data.yaml pointing to the CACHE directories
    # YOLO allows absolute paths in data.yaml.
    
    cache_dir_abs = os.path.abspath(cache_dir)
    
    train_path = os.path.join(cache_dir_abs, "train")
    val_path = os.path.join(cache_dir_abs, "validation")
    test_path = os.path.join(cache_dir_abs, "test") if "test" in splits else None
    
    data_yaml = {
        "train": train_path,
        "val": val_path,
        "nc": len(class_names),
        "names": class_names
    }
    
    if test_path:
        data_yaml["test"] = test_path
    
    # We save data.yaml in the OUTPUT directory for this specific run, 
    # but it points to the shared cache.
    yaml_path = os.path.join(output_dir, "data.yaml")
    
    # Ensure output_dir exists before writing data.yaml
    os.makedirs(output_dir, exist_ok=True)
    
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False)
        
    print(f"Created data.yaml at {yaml_path} pointing to cached data.")
    return yaml_path, class_names
