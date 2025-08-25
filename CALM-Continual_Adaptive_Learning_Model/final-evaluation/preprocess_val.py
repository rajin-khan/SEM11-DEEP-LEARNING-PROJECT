#run command:  python preprocess_val.py --imagenet_root D:\Datasets\imagenet_torchvision --kaggle_root D:\Datasets\imagenet_torchvision

import os
import pandas as pd
from tqdm import tqdm
import argparse
import shutil

def preprocess_validation_set(imagenet_root, original_kaggle_root):
    """
    Sorts the ImageNet validation images into class-specific subfolders.

    Args:
        imagenet_root (str): The path to the reorganized 'imagenet_torchvision' directory.
        original_kaggle_root (str): The path to the original 'imagenet' download directory
                                     which contains LOC_val_solution.csv.
    """
    val_dir = os.path.join(imagenet_root, 'val')
    if not os.path.isdir(val_dir):
        print(f"Error: Validation directory not found at {val_dir}")
        return

    solution_file = os.path.join(original_kaggle_root, 'LOC_val_solution.csv')
    if not os.path.isfile(solution_file):
        print(f"Error: Validation solution file not found at {solution_file}")
        return

    print("Reading validation solutions...")
    df = pd.read_csv(solution_file)
    # Example prediction string: 'n02124075 2 1 1 1' -> we only need the class id 'n02124075'
    df['class_id'] = df['PredictionString'].apply(lambda x: x.split(' ')[0])

    print(f"Found {len(df)} validation images to sort.")
    
    # Create class subdirectories and move images
    for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Sorting validation images"):
        image_name = row['ImageId'] + '.JPEG'
        class_id = row['class_id']
        
        # Create the subdirectory if it doesn't exist
        class_dir = os.path.join(val_dir, class_id)
        os.makedirs(class_dir, exist_ok=True)
        
        # Move the image file
        src_path = os.path.join(val_dir, image_name)
        dest_path = os.path.join(class_dir, image_name)
        
        if os.path.exists(src_path):
            shutil.move(src_path, dest_path)

    print("Validation set preprocessing complete.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Preprocess ImageNet Validation Set")
    parser.add_argument('--imagenet_root', type=str, required=True,
                        help="Path to the reorganized 'imagenet_torchvision' directory.")
    parser.add_argument('--kaggle_root', type=str, required=True,
                        help="Path to the original Kaggle download containing solution files.")
    args = parser.parse_args()
    
    preprocess_validation_set(args.imagenet_root, args.kaggle_root)