import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ChessModel, board_to_tensor
from supervised_dataset_extraction import load_puzzle_data, process_puzzles
import chess
import time
import logging
from tqdm import tqdm
import os
torch.set_num_threads(2) 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessPuzzleDataset(Dataset):
    def __init__(self, processed_data):
        self.data = processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        puzzle = self.data[idx]
        board_tensor = puzzle['board_tensor']
        move = puzzle['moves'][0]  # Get the first move (player's move)
        
        # Convert move to target format
        try:
            chess_move = chess.Move.from_uci(move)
            from_square = chess_move.from_square
            to_square = chess_move.to_square
            target = from_square * 64 + to_square
        except ValueError:
            # If the move is invalid, skip this puzzle
            return self.__getitem__((idx + 1) % len(self))

        return board_tensor, target

def train_model(model, train_loader, criterion, optimizer, device, start_epoch, num_epochs=20):
    model.train()
    total_batches = len(train_loader)
    
    for epoch in range(start_epoch, num_epochs):
        running_loss = 0.0
        start_time = time.time()
        progress_bar = tqdm(total=total_batches, desc="Epoch {}".format(epoch + 1))
        
        for i, (boards, targets) in enumerate(train_loader):
            boards, targets = boards.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': '{:.4f}'.format(loss.item())})
            progress_bar.update()  # Update the progress bar
            
            if (i + 1) % 50 == 0:
                logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Batch [{i + 1}/{total_batches}], '
                             f'Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / total_batches
        epoch_time = time.time() - start_time
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f} seconds. '
                     f'Average Loss: {epoch_loss:.4f}')
        
        # Save model and optimizer state after each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
        }, 'latest_checkpoint.pth')

def main():
    logging.info("Starting the training process...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    csv_file_path = 'supervised_dataset/lichess_db_puzzle.csv'
    logging.info(f"Loading puzzle data from {csv_file_path}")
    puzzles = load_puzzle_data(csv_file_path)
    logging.info(f"Loaded {len(puzzles)} puzzles")
    
    processed_data, skipped_puzzles = process_puzzles(puzzles)
    logging.info(f"Processed {len(processed_data)} puzzles. Skipped {skipped_puzzles} puzzles.")
    
    dataset = ChessPuzzleDataset(processed_data)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)
    logging.info(f"Created DataLoader with {len(train_loader)} batches")
    
    model = ChessModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    start_epoch = 0
    checkpoint_path = 'latest_checkpoint.pth'
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
    
    logging.info("Starting model training...")
    train_model(model, train_loader, criterion, optimizer, device, start_epoch)
    
    logging.info("Training completed. Saving model...")
    torch.save(model.state_dict(), 'supervised_model.pth')
    logging.info("Model saved as 'supervised_model.pth'")

if __name__ == '__main__':
    main()