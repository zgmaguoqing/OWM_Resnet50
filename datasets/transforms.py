import torchvision
from torchvision import transforms

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def val_transforms_l(resize=256, crop_size=224):
    transforms_ = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms_

def test_transforms_l(resize=256, crop_size=224):
    transforms_ = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms_


def train_transforms_l(resize=256, crop_size=224):
    transforms_ = transforms.Compose([
        transforms.Resize((resize, resize)),
        transforms.RandomCrop(crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transforms_
