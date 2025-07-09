import os
import shutil
import random

def copy_half_images(source_dir, baseline_subdir):
    """
    Copies half of the images in each class subdirectory from source_dir to baseline_subdir.
    """
    # Full path to the new baseline subdirectory
    baseline_dir = os.path.join(source_dir, baseline_subdir)
    os.makedirs(baseline_dir, exist_ok=True)

    # Iterate over each class folder (e.g., Positive, Negative)
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        if class_name == baseline_subdir:
            continue  # skip the new baseline folder itself

        baseline_class_path = os.path.join(baseline_dir, class_name)
        os.makedirs(baseline_class_path, exist_ok=True)

        # Get list of images
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

        # Shuffle and select half
        random.shuffle(images)
        num_to_copy = len(images) // 2
        images_to_copy = images[:num_to_copy]

        # Copy images
        for img_name in images_to_copy:
            src = os.path.join(class_path, img_name)
            dst = os.path.join(baseline_class_path, img_name)
            shutil.copy2(src, dst)

        print(f"Copied {num_to_copy} images from '{class_name}' to baseline.")

if __name__ == "__main__":
    source_dir = "."
    baseline_subdir = "baseline"

    copy_half_images(source_dir, baseline_subdir)
    print("✅ Baseline dataset created successfully in 'data/baseline'!")

