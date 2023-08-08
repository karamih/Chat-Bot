import torch
import torch.nn as nn

from model import ChatBotModel
from prepare_data import prepare_data
from create_dataset import create_data_loader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model,
          dataloader,
          loss_fn,
          optimizer,
          epochs,
          device=device):

    for epoch in range(epochs):
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            pred = model(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch+1) % 100 == 0:
            print(f"epoch: {epoch+1}/{epochs}   loss: {loss.item():.4f}")


X_train, y_train, all_words, tags = prepare_data("data.json")

input_size = X_train.shape[1]
output_size = len(tags)
batch_size = 6
learning_rate = 0.001

train_loader = create_data_loader(X_train, y_train, batch_size)

model = ChatBotModel(input_size, output_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), learning_rate)
loss_fn = nn.CrossEntropyLoss()

train(model=model, dataloader=train_loader, optimizer=optimizer, loss_fn=loss_fn, epochs=300)

information_stored = {
    "model_state_dict": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "all_words": all_words,
    "tags": tags
}

file_path = "information_stored.pth"

torch.save(information_stored, file_path)