import torch
from torchvision import transforms as T
from torchvision import datasets
from torch.utils.data import DataLoader, random_split

def get_dataloaders(name, dim, batch_size=256, selected_classes=None, val_ratio=0.1):
    """
    Create and return train, validation, and test dataloaders for a given dataset.

    Args:
        name (str): Name of the dataset ('mnist', 'fer2013' or 'celeba').
        dim (int): Final image dimension (images are resized to [dim, dim]).
        batch_size (int, optional): Batch size for dataloaders.
        selected_classes (list[int], optional): List of class indices to filter the dataset.
        val_ratio (float, optional): Fraction of the training data to use for validation.

    Returns:
        tuple: (train_dataloader, val_dataloader, test_dataloader)
    """
    assert name in ['mnist', 'fer2013', 'celeba'], \
        "Invalid dataset. Available options are: 'mnist', 'fer2013', 'celeba'"

    # Load dataset with appropriate transforms
    if name == 'mnist':
        # MNIST: grayscale images, resized to (dim x dim)
        transform = T.Compose([
            T.ToTensor(),
            T.Resize([dim, dim])              
        ])
        
        training_data = datasets.MNIST(
            root="data",
            train=True,
            download=True,
            transform=transform
        )
        test_data = datasets.MNIST(
            root="data",
            train=False,
            download=True,
            transform=transform
        )

    elif name == 'celeba':
        transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.Lambda(lambda x: x[..., 40:, :]), # hardocded crop to 178x178
            T.Resize([dim, dim])              
        ])
        all_data = datasets.ImageFolder('data/CelebA/img_align_celeba', transform=transform)

        training_data, test_data = random_split(all_data, [0.8, 0.2])

    elif name == 'fer2013':
        transform = T.Compose([
            T.ToTensor(),
            T.Grayscale(),
            T.Resize([dim, dim])
        ])
        training_data = datasets.ImageFolder('data/fer2013/train', transform=transform)
        test_data = datasets.ImageFolder('data/fer2013/test', transform=transform)

    # Filter by selected classes (if provided)
    if selected_classes is not None:
        tensor_select = torch.tensor(selected_classes)

        # Ensure targets are tensors (ImageFolder uses list)
        train_targets = training_data.targets
        if not torch.is_tensor(train_targets):
            train_targets = torch.tensor(train_targets)

        test_targets = test_data.targets
        if not torch.is_tensor(test_targets):
            test_targets = torch.tensor(test_targets)

        # Get indices of samples belonging to selected classes
        train_indices = torch.where(torch.isin(train_targets, tensor_select))[0]
        test_indices = torch.where(torch.isin(test_targets, tensor_select))[0]

        # Create subsets containing only the selected classes
        train_subset = torch.utils.data.Subset(training_data, train_indices)
        test_subset = torch.utils.data.Subset(test_data, test_indices)

        # Map original class labels -> new labels starting from 0
        class_map = {orig: new for new, orig in enumerate(selected_classes)}

        # Final filtered datasets as list of tuples (image, remapped_label)
        training_data = [(x, class_map[int(y)]) for x, y in train_subset]
        test_data = [(x, class_map[int(y)]) for x, y in test_subset]

    train_set, val_set = random_split(training_data, [1 - val_ratio, val_ratio])

    # Create DataLoaders
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=32)
    val_dataloader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=32)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=32)

    return train_dataloader, val_dataloader, test_dataloader
