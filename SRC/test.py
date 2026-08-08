import torch
import torch.optim as optim
import torch.nn as nn

from model import Net


def train(model, X_train, Y_train, epochs=4, batch_size=1000):
    optimizer = optim.Adam(
        model.parameters(),
        lr=0.001,
        betas=(0.9, 0.999),
        weight_decay=1e-5
    )

    loss_function = nn.CrossEntropyLoss()
    model.train()

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            batch_X = X_train[i:i + batch_size].view(-1, 2, 128)
            batch_y = Y_train[i:i + batch_size].long()

            output = model(batch_X)
            loss = loss_function(output, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item():.4f}")

    return model


if __name__ == "__main__":
    X_train = torch.load("data/X_train.pt")
    Y_train = torch.load("data/Y_train.pt")

    model = Net()
    model = train(model, X_train, Y_train)

    torch.save(model.state_dict(), "models/AMC_model.pt")
    print("Model saved.")
