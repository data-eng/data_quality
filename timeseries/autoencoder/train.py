import time
import warnings
from tqdm import tqdm
import matplotlib.pyplot as plt

from .. import utils
from ..loader import *
from .model import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f'Device is {device}')

warnings.filterwarnings("ignore", category=FutureWarning)

def train(data, epochs, patience, lr, criterion, model, optimizer, scheduler, max_grad_norm=1.0, visualize=False):
    model.to(device)

    train_data, val_data = data
    batches = len(train_data)

    logger.info(f"Number of training iterations per epoch: {batches}")

    optimizer = utils.get_optim(optimizer, model, lr)
    scheduler = utils.get_sched(*scheduler, optimizer)

    train_time = 0.0
    best_val_loss = float('inf')
    stationary = 0
    train_losses, val_losses = [], []

    checkpoints = {
        'epochs': 0, 
        'best_epoch': 0, 
        'best_train_loss': float('inf'), 
        'best_val_loss': float('inf'),
        'train_time': 0.0 
    }

    for epoch in range(epochs):
        start = time.time()
        total_train_loss = 0.0

        model.train()

        #progress_bar = tqdm(enumerate(train_data), total=batches, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)
        
        for _, (X, _) in enumerate(train_data):
        #for _, (X, _) in progress_bar:
            X = X.to(device)

            X, _ = separate(src=X, c=[0,1], t=[3])
            X_dec, _ = model(X)

            train_loss = criterion(X_dec, X)
            optimizer.zero_grad()
            train_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            total_train_loss += train_loss.item()
            #progress_bar.set_postfix(Loss=train_loss.item())

        avg_train_loss = total_train_loss / batches
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for _, (X, _) in enumerate(val_data):
                X = X.to(device)

                X, _ = separate(src=X, c=[0,1], t=[3])
                X_dec, _ = model(X)

                val_loss = criterion(X_dec, X)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / batches
        val_losses.append(avg_val_loss)

        end = time.time()
        duration = end - start
        train_time += duration

        logger.info(f'Epoch [{epoch + 1}/{epochs}], Training Loss: {avg_train_loss:.6f}, Validation Loss: {avg_val_loss:.6f}, Duration: {duration:.2f}s')

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

    checkpoints.update({
        'epochs': epoch + 1,
        'train_time': train_time})
    
    cfn = utils.get_path('static', 'autoencoder', filename='train_checkpoints.json')
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

    samples, chunks = 7680, 32
    seq_len = samples // chunks

    datapaths = split_data(dir=npz_dir, train_size=57, val_size=1, test_size=1)
    
    train_df, val_df, _ = get_dataframes(datapaths, rate=seq_len, exist=True)

    datasets = create_datasets(dataframes=(train_df, val_df), seq_len=seq_len)

    dataloaders = create_dataloaders(datasets, batch_size=512, drop_last=False)

    model = Autoencoder(seq_len=240,
                        num_feats=2, 
                        latent_seq_len=1, 
                        latent_num_feats=8, 
                        hidden_size=8, 
                        num_layers=1,
                        dropout=0.5)
    
    train(data=dataloaders,
          epochs=1000,
          patience=30,
          lr=5e-4,
          criterion=utils.PNormLoss(p=2),
          model=model,
          optimizer='AdamW',
          scheduler=('StepLR', 1.0, 0.98),
          max_grad_norm=1.0,
          visualize=True)

if __name__ == '__main__':
    main()