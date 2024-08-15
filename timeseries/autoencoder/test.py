import torch
from tqdm import tqdm
import warnings
import matplotlib.pyplot as plt
import librosa
import librosa.display

from .. import utils
from ..loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def plot_spectrograms(signals, fs):
    all_signals = []

    def design_spectrogram(signal, fs, n_fft, title, figsize):
        signal = signal.cpu().numpy()

        batch_size, seq_len, num_feats = signal.shape
        signal_concat = signal.reshape(batch_size * seq_len, num_feats)

        fig, axes = plt.subplots(1, num_feats, figsize=figsize)
        fig.suptitle(f'{title.capitalize()} Signal Spectrogram', fontsize=16)
        axes = axes.flatten()

        for i in range(num_feats):
            S = librosa.feature.melspectrogram(y=signal_concat[:, i], sr=fs, n_mels=128, n_fft=n_fft)
            S_dB = librosa.power_to_db(S, ref=np.max)

            librosa.display.specshow(S_dB, sr=fs, ax=axes[i], x_axis='time', y_axis='mel')
            axes[i].set(title=f'Channel {i+1}')
            axes[i].set_yticks([tick for tick in axes[i].get_yticks() if tick != 0.0])

        for j in range(num_feats, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()

        path = utils.get_path('static', 'autoencoder', filename=f'{title}_spectro.png')
        fig.savefig(path)
        plt.close(fig)

    for _, (X, X_enc) in enumerate(signals):
        all_signals.append((X, 'raw', 122880//3072, (10, 5)))
        all_signals.append((X_enc, 'encoded', 512//1, (80, 5)))

    for signal, title, n_fft, figsize in all_signals:
        design_spectrogram(signal, fs, n_fft, title, figsize)

def test(data, criterion, model, fs, visualize=False):
    mfn = utils.get_path('models', filename='autoencoder.pth')

    model.load_state_dict(torch.load(mfn))
    model.to(device)
    model.eval()

    batches = len(data)
    total_test_loss = 0.0
    signals = []

    progress_bar = tqdm(enumerate(data), total=batches, desc=f'Evaluation', leave=True)

    with torch.no_grad():
        for _, (X, _) in progress_bar:
            X = X[:, :, :2].to(device)

            X_dec, latent = model(X)

            test_loss = criterion(X_dec, X)

            total_test_loss += test_loss.item()
            progress_bar.set_postfix(Loss=test_loss.item())

            signals.append((X, latent))

        if visualize:
            plot_spectrograms(signals, fs)

        avg_test_loss = total_test_loss / batches

    logger.info(f'\nTesting complete!\nTesting Loss: {avg_test_loss:.6f}\n')

def main():
    npz_dir = utils.get_dir('data', 'npz')
    
    samples, chunks = 7680, 32
    seq_len = samples // chunks
    
    datapaths = split_data(dir=npz_dir, train_size=57, val_size=1, test_size=1)
    
    _, _, test_df = get_dataframes(datapaths, rate=seq_len, exist=True)

    datasets = create_datasets(dataframes=(test_df,), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512, drop_last=True)

    model = Autoencoder(seq_len=240,
                        num_feats=2, 
                        latent_seq_len=1, 
                        latent_num_feats=8, 
                        hidden_size=128, 
                        num_layers=2)
    
    test(data=dataloaders[0],
         criterion=nn.MSELoss(),
         model=model,
         fs=get_fs(path=datapaths),
         visualize=True)

if __name__ == '__main__':
    main()