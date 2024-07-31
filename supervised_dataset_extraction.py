import csv
import chess
import torch
from model import board_to_tensor

def load_puzzle_data(*csv_file_paths):
    puzzles = []
    for csv_file_path in csv_file_paths:
        with open(csv_file_path, 'r') as file:
            csv_reader = csv.reader(file)
            for row in csv_reader:
                puzzles.append({
                    'fen': row[1],
                    'moves': row[2].split(),
                    'rating': int(row[3])
                })
    return puzzles

def process_puzzles(puzzles):
    processed_data = []
    skipped_puzzles = 0
    for puzzle in puzzles:
        board = chess.Board(puzzle['fen'])
        
        # Skip puzzles where it's black to move
        if board.turn == chess.BLACK:
            skipped_puzzles += 1
            continue
        
        moves = puzzle['moves']
        tensor = board_to_tensor(board)
        
        processed_moves = []
        for move in moves:
            try:
                chess_move = chess.Move.from_uci(move)
                processed_moves.append(move)
            except ValueError:
                continue  # Skip invalid moves
        
        if processed_moves:
            processed_data.append({
                'board_tensor': tensor,
                'moves': processed_moves,
                'rating': puzzle['rating']
            })
    
    return processed_data, skipped_puzzles

def main():
    train_csv_path = 'supervised_dataset/larger_lichess_db_puzzle.csv'
    val_csv_path = 'supervised_dataset/lichess_puzzle_val.csv'
    test_csv_path = 'supervised_dataset/lichess_puzzle_test.csv'
    
    train_puzzles = load_puzzle_data(train_csv_path)
    val_puzzles = load_puzzle_data(val_csv_path)
    test_puzzles = load_puzzle_data(test_csv_path)
    
    train_data, train_skipped = process_puzzles(train_puzzles)
    val_data, val_skipped = process_puzzles(val_puzzles)
    test_data, test_skipped = process_puzzles(test_puzzles)
    
    print(f"Train puzzles processed: {len(train_data)}, skipped: {train_skipped}")
    print(f"Validation puzzles processed: {len(val_data)}, skipped: {val_skipped}")
    print(f"Test puzzles processed: {len(test_data)}, skipped: {test_skipped}")
    print(f"Puzzles skipped (black to move): {skipped_puzzles}")
    if processed_data:
        print(f"Sample puzzle:")
        print(f"  Board tensor shape: {processed_data[0]['board_tensor'].shape}")
        print(f"  Moves: {processed_data[0]['moves']}")
        print(f"  Rating: {processed_data[0]['rating']}")
    else:
        print("No puzzles were processed. All puzzles might be from black's perspective.")

if __name__ == "__main__":
    main()