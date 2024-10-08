import torch
from torchvision import datasets, transforms

data_dir = 'flowers'
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
def my_transforms(train_dir, valid_dir, test_dir):
    train_transform_pipeline = transforms.Compose([
        transforms.RandomResizedCrop(224),            
        transforms.RandomHorizontalFlip(p=0.5), 
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    test_transform_pipeline = transforms.Compose([
        transforms.Resize(256),                      
        transforms.CenterCrop(224),                
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
    ])

    training_dataset = datasets.ImageFolder(train_dir, transform=train_transform_pipeline)
    validation_dataset = datasets.ImageFolder(valid_dir, transform=test_transform_pipeline)
    testing_dataset = datasets.ImageFolder(test_dir, transform=test_transform_pipeline)

    train_loader = torch.utils.data.DataLoader(training_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(testing_dataset, batch_size=64, shuffle=False)

    return train_loader, valid_loader, test_loader, training_dataset