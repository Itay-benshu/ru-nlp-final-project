import os
import shutil
import random

def split_dataset(root_dir, split_ratio=(0.9, 0.1), random_seed=None):
    train_ratio, test_ratio = split_ratio
    assert train_ratio + test_ratio == 1.0, "Split ratios should add up to 1.0"

    subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]

    if random_seed is not None:
        random.seed(random_seed)
        random.shuffle(subdirectories)

    num_subdirectories = len(subdirectories)
    num_train = int(train_ratio * num_subdirectories)
    num_test = int(test_ratio * num_subdirectories)

    train_directories = subdirectories[:num_train]
    test_directories = subdirectories[num_train:num_train + num_test]
    validation_directories = train_directories[:int(0.1 * len(train_directories))]
    train_directories = train_directories[int(0.1 * len(train_directories)):]

    for split, dirs in [("train", train_directories), ("test", test_directories), ("validation", validation_directories)]:
        split_dir = os.path.join(root_dir, '../', split)
        if os.path.exists(split_dir):
            shutil.rmtree(split_dir)

        os.makedirs(split_dir, exist_ok=True)
        
        for dir_name in dirs:
            src_dir = os.path.join(root_dir, dir_name)
            dst_dir = os.path.join(split_dir, dir_name)
            shutil.copytree(src_dir, dst_dir)