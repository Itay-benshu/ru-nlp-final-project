import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torchvision.transforms import v2 as transforms_v2
import torch

MAX_LENGTH = 32

class SongsDataset(Dataset):
    def __init__(self, root_dir, image_processor, tokenizer, by_line=False, n_variations=1):
        self.root_dir = root_dir
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        self.subdirectories = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        self.by_line = by_line
        self.n_variations = n_variations

        # Last init fn.
        self.data = self._load_data()

    def _load_data(self):
        data = []
        for subdir in self.subdirectories:
            img_path = os.path.join(self.root_dir, subdir, 'spectrogram.png')
            lyrics_path = os.path.join(self.root_dir, subdir, 'chorus.txt' if not self.by_line else 'lyrics.txt')
            
            if os.path.exists(img_path) and os.path.exists(lyrics_path):
                original_img = Image.open(img_path).convert('RGB')
                for _ in range(self.n_variations):
                    
                    all_transforms = [transforms.Resize((224, original_img.height)),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop((224, 224)),
                                    transforms_v2.RandomErasing(p=1, scale=(0.1, 0.5))]
                    
                    transform = transforms.Compose(all_transforms if self.n_variations > 1 else all_transforms[:-1])
                    img = self.image_processor(transform(original_img), return_tensors='pt').pixel_values[0, :, :, :]

                    with open(lyrics_path, 'r') as f:
                        try:
                            lyrics = f.read()
                        except:
                            continue

                    if self.by_line:
                        lines = list(set(lyrics.split('\n')))

                        for line in lines:
                            tokenized_line = self.tokenizer(line, return_tensors="pt", max_length=MAX_LENGTH // 2, 
                                                    padding='max_length', truncation=True).input_ids[0]

                            data.append({
                                'image': img,
                                'lyrics': tokenized_line
                            })
                    else:
                        lyrics = self.tokenizer(lyrics, return_tensors="pt", max_length=MAX_LENGTH, 
                                                        padding='max_length', truncation=True).input_ids[0]

                        data.append({
                            'image': img,
                            'lyrics': lyrics
                        })

        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]