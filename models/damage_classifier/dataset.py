import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ['minor', 'moderate', 'severe']
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

TRAIN_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

VAL_TRANSFORMS = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class CarDamageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Expects: image_dir/minor/*.jpg, image_dir/moderate/*.jpg, etc.
        self.samples = []
        self.transform = transform
        for cls in CLASSES:
            cls_dir = os.path.join(image_dir, cls)
            if not os.path.exists(cls_dir):
                continue
            for fname in os.listdir(cls_dir):
                if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, fname), CLASS_TO_IDX[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label
