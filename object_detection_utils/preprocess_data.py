# object_detection_utils/preprocess_data.py

import argparse
import json
import os
import glob # <-- Added
import collections
import datasets

def get_annotation_file(split_dir: str):
    """Helper to find the first .json file in a directory."""
    if not os.path.exists(split_dir):
        return None
        
    # Find any .json file
    json_files = glob.glob(os.path.join(split_dir, "*.json"))
    # Filter out hidden files
    valid_jsons = [f for f in json_files if not os.path.basename(f).startswith(".")]
    
    if not valid_jsons:
        return None
    
    # Return the first one found
    return valid_jsons[0]

def preprocess_and_save_dataset(raw_data_dir: str, processed_data_dir: str):
    """
    Processes a raw COCO-style dataset and saves it to disk in Arrow format.
    """
    print(f"Starting preprocessing of data from '{raw_data_dir}'")
    
    # Dynamically find annotation files across all splits
    all_categories = {}
    
    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(raw_data_dir, split)
        annot_path = get_annotation_file(split_dir)
        
        if annot_path:
            try:
                with open(annot_path, "r") as f:
                    data = json.load(f)
                    for cat in data.get("categories", []):
                        if cat["id"] not in all_categories:
                            all_categories[cat["id"]] = cat["name"]
            except Exception as e:
                print(f"Warning: Could not read categories from {annot_path}: {e}")
                
    if not all_categories:
        raise RuntimeError(f"Could not find any valid categories in .json annotation files across {raw_data_dir}")
        
    # Sort by ID to ensure consistent mapping
    categories = [all_categories[k] for k in sorted(all_categories.keys())]
    print(f"Aggregated {len(categories)} unique categories across all splits.")

    features = datasets.Features({
        "image_id": datasets.Value("int64"),
        "image": datasets.Image(),
        "width": datasets.Value("int32"),
        "height": datasets.Value("int32"),
        "objects": {
            "id": datasets.Sequence(datasets.Value("int64")),
            "area": datasets.Sequence(datasets.Value("int64")),
            "bbox": datasets.Sequence(datasets.Sequence(datasets.Value("float32"), length=4)),
            "category": datasets.Sequence(datasets.ClassLabel(names=categories)),
        }
    })

    def generate_examples(split: str):
        """Generator function that yields examples for a given split."""
        split_dir = os.path.join(raw_data_dir, split)
        annot_path = get_annotation_file(split_dir)
        
        if not annot_path:
            print(f"Warning: No .json annotation file found for split '{split}'. Skipping.")
            return

        print(f"Processing split '{split}' using annotations: {annot_path}")
        with open(annot_path, "r") as f:
            data = json.load(f)

        category_id_to_name = {cat["id"]: cat["name"] for cat in data["categories"]}
        image_id_to_annotations = collections.defaultdict(list)
        for annot in data["annotations"]:
            image_id_to_annotations[annot["image_id"]].append(annot)
        
        for image_info in data["images"]:
            # The image file is usually in the same folder as the json
            image_path = os.path.join(split_dir, image_info["file_name"])
            
            # Fallback: sometimes images are in an 'images' subfolder? 
            # For now, we assume standard Roboflow structure (flat in split dir)
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue

            image_id = image_info["id"]
            annotations = image_id_to_annotations[image_id]

            objects = {
                "id": [annot["id"] for annot in annotations],
                "area": [annot["area"] for annot in annotations],
                "bbox": [annot["bbox"] for annot in annotations],
                "category": [category_id_to_name[annot["category_id"]] for annot in annotations],
            }
            
            yield {
                "image_id": image_id,
                "image": image_path,
                "width": image_info["width"],
                "height": image_info["height"],
                "objects": objects,
            }

    dataset_dict = datasets.DatasetDict()
    for split in ["train", "validation", "test"]:
        if os.path.exists(os.path.join(raw_data_dir, split)):
            dataset_dict[split] = datasets.Dataset.from_generator(
                generate_examples,
                gen_kwargs={"split": split},
                features=features
            )

    print(f"Saving processed dataset to '{processed_data_dir}'...")
    dataset_dict.save_to_disk(processed_data_dir)
    print("Preprocessing complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_data_dir", type=str, required=True)
    parser.add_argument("--processed_data_dir", type=str, required=True)
    args = parser.parse_args()
    
    preprocess_and_save_dataset(args.raw_data_dir, args.processed_data_dir)