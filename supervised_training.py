import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from model import ChessModel, board_to_tensor, get_lr_scheduler
from supervised_dataset_extraction import load_puzzle_data, process_puzzles
import chess
import time
import logging
from tqdm import tqdm
import os
torch.set_num_threads(2) 

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class ChessPuzzleDataset(Dataset):
    def __init__(self, processed_data, start_idx=0):
        self.full_data = processed_data
        self.reset(start_idx)

    def reset(self, start_idx=0):
        self.data = self.full_data[start_idx:]
        self.start_idx = start_idx

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

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, start_epoch, start_batch, global_batch, num_epochs=5, patience=3):
    model.train()
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    
    for epoch in range(start_epoch, num_epochs):
        if epoch > start_epoch:
            train_loader.dataset.reset()
            start_batch = 0
        
        running_loss = 0.0
        start_time = time.time()
        progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}", initial=start_batch)
        
        for i, (boards, targets) in enumerate(train_loader, start=start_batch):
            if i < start_batch:
                continue
            
            boards, targets = boards.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            progress_bar.update()
            
            global_batch += 1
            
            if global_batch % 100 == 0:
                torch.save({
                    'epoch': epoch,
                    'batch': i + 1,
                    'global_batch': global_batch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'loss': loss.item(),
                }, f'checkpoint_epoch_{epoch}_batch_{i+1}.pth')
                logging.info(f"Saved checkpoint at epoch {epoch + 1}, batch {i + 1}")
        
        if len(train_loader) > 0:
            epoch_loss = running_loss / (len(train_loader) - start_batch)
        else:
            epoch_loss = 0
        epoch_time = time.time() - start_time
        
        # Validate the model
        val_loss, val_accuracy = validate_model(model, val_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        logging.info(f'Epoch [{epoch + 1}/{num_epochs}] completed in {epoch_time:.2f} seconds. '
                     f'Train Loss: {epoch_loss:.4f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Save epoch checkpoint
        torch.save({
            'epoch': epoch + 1,
            'batch': 0,
            'global_batch': global_batch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': epoch_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }, f'checkpoint_epoch_{epoch + 1}.pth')
        logging.info(f"Saved epoch checkpoint: checkpoint_epoch_{epoch + 1}.pth")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save({
                'epoch': epoch + 1,
                'batch': 0,
                'global_batch': global_batch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }, 'best_model_checkpoint.pth')
        else:
            epochs_without_improvement += 1
        
        if epochs_without_improvement >= patience:
            logging.info(f'Early stopping triggered after {epoch + 1} epochs')
            break
        
        start_batch = 0  # Reset start_batch for the next epoch
    
    # Load the best model
    best_checkpoint = torch.load('best_model_checkpoint.pth')
    model.load_state_dict(best_checkpoint['model_state_dict'])
    return model
   
def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for boards, targets in val_loader:
            boards, targets = boards.to(device), targets.to(device)
            outputs = model(boards)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def test_model(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for boards, targets in test_loader:
            boards, targets = boards.to(device), targets.to(device)
            outputs = model(boards)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    avg_loss = total_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

def main():
    logging.info("Starting the training process...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    train_csv_path = 'supervised_dataset/larger_lichess_db_puzzle.csv'
    val_csv_path = 'supervised_dataset/lichess_puzzle_val.csv'
    test_csv_path = 'supervised_dataset/lichess_puzzle_test.csv'
    
    logging.info("Loading and processing puzzle data...")
    train_puzzles = load_puzzle_data(train_csv_path)
    val_puzzles = load_puzzle_data(val_csv_path)
    test_puzzles = load_puzzle_data(test_csv_path)
    
    train_data, train_skipped = process_puzzles(train_puzzles)
    val_data, val_skipped = process_puzzles(val_puzzles)
    test_data, test_skipped = process_puzzles(test_puzzles)
    
    logging.info(f"Train: processed {len(train_data)}, skipped {train_skipped}")
    logging.info(f"Validation: processed {len(val_data)}, skipped {val_skipped}")
    logging.info(f"Test: processed {len(test_data)}, skipped {test_skipped}")
    
    train_dataset = ChessPuzzleDataset(train_data)
    val_dataset = ChessPuzzleDataset(val_data)
    test_dataset = ChessPuzzleDataset(test_data)
    
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    
    logging.info(f"Created DataLoaders: Train: {len(train_loader)} batches, Val: {len(val_loader)} batches, Test: {len(test_loader)} batches")
    
    model = ChessModel(dropout_rate=0.7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = get_lr_scheduler(optimizer)
    
    start_epoch = 0
    start_batch = 0
    global_batch = 0
    
    # Find the latest checkpoint
    checkpoints = [f for f in os.listdir() if f.startswith('checkpoint_') and f.endswith('.pth')]
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=os.path.getctime)
        checkpoint_path = latest_checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
        start_batch = checkpoint['batch']
        global_batch = checkpoint['global_batch']
        logging.info(f"Resuming from checkpoint: {checkpoint_path}")
        logging.info(f"Starting from epoch {start_epoch + 1}, batch {start_batch}")
        
        # Calculate the starting index for the dataset
        start_idx = global_batch * train_loader.batch_size
        
        # Create new DataLoader starting from the correct position
        train_dataset = ChessPuzzleDataset(train_data, start_idx=start_idx)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=False)
    else:
        train_dataset = ChessPuzzleDataset(train_data)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    logging.info("Starting model training...")
    model = train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, start_epoch, start_batch, global_batch)

    
    logging.info("Training completed. Saving model...")
    torch.save(model.state_dict(), 'supervised_model.pth')
    logging.info("Model saved as 'supervised_model.pth'")
    
    # Test the model
    test_loss, test_accuracy = test_model(model, test_loader, criterion, device)
    logging.info(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")

if __name__ == '__main__':
    main()