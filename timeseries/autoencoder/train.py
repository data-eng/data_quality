import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

from .. import utils
from .loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

def train(data, epochs, patience, lr, criterion, model, optimizer, scheduler, visualize=False):
    model.to(device)

    train_data, val_data = data

    batches = len(train_data)

    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)

    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    checkpoints = {
        'epochs': 0, 
        'best_epoch': 0, 
        'best_train_loss': float('inf'), 
        'best_val_loss': float('inf')
    }

    for epoch in range(epochs):
        model.train()

        total_train_loss = 0.0

        progress_bar = tqdm(enumerate(train_data), total=batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

        for _, (X, _) in progress_bar:
            X = X[:, :, :2].to(device)

            X_dec, _ = model(X)

            train_loss = criterion(X_dec, X)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            total_train_loss += train_loss.item()
            progress_bar.set_postfix(Loss=train_loss.item())

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for X, _ in val_data:
                X = X[:, :, :2].to(device)

                X_dec, _ = model(X)

                val_loss = criterion(X_dec, X)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        logger.info(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}')

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            stationary = 0

            logger.info(f'New best val found! ~ Epoch [{epoch + 1}/{epochs}], Val Loss {avg_val_loss}')

            path = utils.get_path('models', filename='autoencoder.pth')
            torch.save(model.state_dict(), path)

            checkpoints.update({
                'best_epoch': epoch + 1, 
                'best_train_loss': avg_train_loss, 
                'best_val_loss': best_val_loss
            })
                
        else:
            stationary += 1

        if stationary >= patience:
            logger.info(f'Early stopping after {epoch + 1} epochs without improvement. Patience is {patience}.')
            break

        scheduler.step()

    cfn = utils.get_path('static', 'autoencoder', filename='train_checkpoints.json')
    checkpoints.update({'epochs': epoch + 1})
    utils.save_json(data=checkpoints, filename=cfn)
    
    if visualize:
        utils.visualize(type='multi-plot',
                        values=[(range(1, len(train_losses) + 1), train_losses), (range(1, len(val_losses) + 1), val_losses)], 
                        labels=('Epoch', 'Loss'), 
                        title='Loss Curves',
                        plot_func=plt.plot,
                        coloring=['brown', 'royalblue'],
                        names=['Training', 'Validation'],
                        path=utils.get_dir('static', 'autoencoder'))

    logger.info(f'\nTraining complete!\nFinal Training Loss: {avg_train_loss:.6f} & Validation Loss: {best_val_loss:.6f}\n')

def main():
    npz_dir = utils.get_dir('data', 'npz')
    seq_len = 7680 // 32
    datapaths = split_data(dir=npz_dir, train_size=57, val_size=1, test_size=1)
    
    train_df, val_df, _ = get_dataframes(datapaths, exist=False)

    datasets = create_datasets(dataframes=(train_df, val_df), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512)

    model = Autoencoder(seq_len=240,
                        num_feats=2, 
                        latent_seq_len=1, 
                        latent_num_feats=8, 
                        hidden_size=128, 
                        num_layers=2)
    
    train(data=dataloaders,
          epochs=2,
          patience=30,
          lr=5e-4,
          criterion=nn.MSELoss(),
          model=model,
          optimizer='AdamW',
          scheduler=('StepLR', 1.0, 0.98),
          visualize=True)

if __name__ == '__main__':
    main()