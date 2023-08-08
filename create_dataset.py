from torch.utils.data import DataLoader, Dataset


class ChatBotDataset(Dataset):
    def __init__(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        self.n_samples = len(X_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]

    def __len__(self):
        return self.n_samples


def create_data_loader(X_train, y_train, batch_size):
    dataset = ChatBotDataset(X_train, y_train)

    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return train_loader
