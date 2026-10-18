import os
import shutil
import random

SOURCE_DIR = "train"
DEST_DIR="dataset"

TRAIN_RATIO=0.8

classes=["Closed_Eyes","Open_Eyes"]

for class_name in classes:

    source_folder=os.path.join(SOURCE_DIR,class_name)

    train_folder=os.path.join(DEST_DIR,"test",class_name )
    test_folder = os.path.join(DEST_DIR, "test", class_name)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    images = os.listdir(source_folder)

    random.shuffle(images)

    split_index = int(len(images) * TRAIN_RATIO)

    train_images = images[:split_index]
    test_images = images[split_index:]

    for image in train_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(train_folder, image)
        )

    for image in test_images:
        shutil.copy(
            os.path.join(source_folder, image),
            os.path.join(test_folder, image)
        )

    print(
        f"{class_name}: "
        f"{len(train_images)} train, "
        f"{len(test_images)} test"
    )

print("Dataset split complete!")