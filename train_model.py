import torch
import torchaudio
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from cnn_model import CNN
from urbansounddataset import UrbanSoundDataset
from datetime import datetime
import time


ANNOTATION_FILE = "./dataset/UrbanSound8K.csv"
AUDIO_DIR = "./dataset/audio"
MODEL_DIR = "./model/"
SAMPLE_RATE = 22050
NUM_SAMPLES = 22050

# Hyperparameters
BATCH_SIZE = 128
EPOCHS = 5
LEARNING_RATE = 0.001


def create_data_loader(data, batch_size, shuffle=False, num_workers=0):
    """ Create a data loader
    Args:
        data (Dataset): dataset
        batch_size (int): batch size
        shuffle (bool): whether to shuffle the data at every epoch
        num_workers (int): number of subprocesses for data loading
    Returns
    -------
    DataLoader
        a dataset loader
    """
    return DataLoader(data, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train the model for 1 epoch
    Args:
        model: nn.Module
        train_loader: train DataLoader
        criterion: callable loss function
        optimizer: pytorch optimizer
        device: torch.device
    Returns
    -------
    Tuple[Float, Float]
        average train loss and average train accuracy for current epoch
    """

    train_losses = []
    train_corrects = []
    model.train()

    # print(len(train_loader.dataset)) # 8732
    # print(len(train_loader)) # 69

    # Iterate over data.
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # prediction
        outputs = model(inputs)

        # calculate loss
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)

        # backprop
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # statistics
        train_losses.append(loss.item())
        train_corrects.append(torch.sum(preds == labels.data).item())

    return sum(train_losses)/len(train_losses), sum(train_corrects)/len(train_loader.dataset)


def val_epoch(model, val_loader, criterion, device):
    """Validate the model for 1 epoch
    Args:
        model: nn.Module
        val_loader: val DataLoader
        criterion: callable loss function
        device: torch.device

    Returns
    -------
    Tuple[Float, Float]
        average val loss and average val accuracy for current epoch
    """

    val_losses = []
    val_corrects = []
    model.eval()

    # Iterate over data
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # prediction
            outputs = model(inputs)

            # calculate loss
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # statistics
            val_losses.append(loss.item())
            val_corrects.append(torch.sum(preds == labels.data).item())

    return sum(val_losses)/len(val_losses), sum(val_corrects)/len(val_loader.dataset)


if __name__ == "__main__":
    current_time = datetime.now().strftime('%b%d_%H-%M-%S')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # create transform & dataset
    mel_spectogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )

    dataset = UrbanSoundDataset(
        ANNOTATION_FILE,
        AUDIO_DIR,
        mel_spectogram,
        SAMPLE_RATE,
        NUM_SAMPLES,
        device
    )

    # create train data loaders
    train_loader = create_data_loader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # instantiate model
    cnn = CNN().to(device)

    # loss, optimizer and scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(cnn.parameters(), lr=LEARNING_RATE)

    # training
    print("Start training...")
    since = time.time()
    for epoch in range(EPOCHS):
        # train
        train_loss, train_acc = train_epoch(cnn, train_loader, criterion, optimizer, device)
        message = f'Epoch: {epoch + 1}/{EPOCHS} \tTrainLoss: {train_loss:.4f} \tTrainAcc: {train_acc:.4f}'
        print(message)
    time_elapsed = time.time() - since
    print(f"Training complete in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")

    # Save model
    try:
        torch.save(cnn.state_dict(), MODEL_DIR + "model.pt")
        print(f"Trained model is saved to {MODEL_DIR} as model.pt")
    except Exception as e:
        print(e)
        try:
            torch.save(cnn.state_dict(), "model.pt")
            print(f"Trained model is saved to current work directory as model.pt")
        except Exception as e:
            print(e)
            print("Saving model failed")



