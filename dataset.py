from torch.utils.data import Dataset
import torchvision.transforms as transforms

class PatternDataset(Dataset):
    def __init__(self, full_image, image_size, sample_cnt) -> None:
        super().__init__()
        self.img = full_image
        self.img_trans = transforms.Compose([
            transforms.RandomCrop(image_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2 - 1)
        ])
        self.sample_cnt = sample_cnt
    
    def __getitem__(self, index):
        return self.img_trans(self.img)

    def __len__(self):
        return self.sample_cnt