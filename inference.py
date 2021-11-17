import torch
import torchaudio
from cnn_model import CNN
from urbansounddataset import UrbanSoundDataset
from train_model import AUDIO_DIR, ANNOTATION_FILE, SAMPLE_RATE, NUM_SAMPLES, MODEL_DIR


class_mapping = [
    "air_conditioner",
    "car_horn",
    "children_playing",
    "dog_bark",
    "drilling",
    "engine_idling",
    "gun_shot",
    "jackhammer",
    "siren",
    "street_music"
]


def predict(model, input, target, class_mapping):
    model.eval()
    with torch.no_grad():
        output = model(input)
        # etc Tensor (1, 10) -> [ [0.1, 0.01, ..., 0.6] ]
        _, pred_index = torch.max(output, dim=1)
        predicted = class_mapping[pred_index]
        expected = class_mapping[target]
        return predicted, expected


if __name__ == '__main__':
    # device
    device = torch.device('cpu')

    # load back the model
    cnn = CNN()
    cnn.load_state_dict(torch.load(MODEL_DIR + 'model.pt', map_location=device))

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

    # get one sample from dataset for inference
    index = 0
    input, target = dataset[index][0], dataset[index][1]
    input.unsqueeze_(0)  # for cnn input as it is 4 dim (batch_size. channels, frequency, time)

    # make an inference
    predicted, expected = predict(cnn, input, target,
                                  class_mapping)
    print(f"Predicted: '{predicted}', expected: '{expected}'")
