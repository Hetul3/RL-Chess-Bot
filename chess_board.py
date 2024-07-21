import chess
import chess.engine
import numpy as np
import torch

# printing the legal moves
def print_legal_moves(board):
    print("Legal moves:")
    for move in board.legal_moves:
        print(move, end=" ")
    print()
    
def board_to_tensor(board):
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    # tensor, represents all 12 pieces with a board for each piece showing where they are
    tensor = torch.zeros(12, 8, 8)
    for i, piece in enumerate(pieces):
        for square in board.pieces(chess.PIECE_SYMBOLS.index(piece.lower()), piece.isupper()):
            rank, file = divmod(square, 8)
            tensor[i][rank][file] = 1
            
    return tensor
    
def main():
    #setting up board and stockfish
    board = chess.Board()
    stockfish_path = "stockfish/stockfish"
    stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    
    while not board.is_game_over():
        print(board)
        print_legal_moves(board)
        
        user_move = input("Enter your move: ")
        try:
            board.push_san(user_move)
        except ValueError:
            print("Invalid move. Try again.")
            continue
        
        result = stockfish.play(board, chess.engine.Limit(time=0.1))
        board.push(result.move)
        
        print("\nStockfish move:", result.move)
        print(board)
        print("\n")
    
    engine.quit()
    
if __name__ == "__main__":
    main()
    
