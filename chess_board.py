import chess
import chess.engine
from IPython.display import SVG, display
import chess.svg

# print in an aesthetic way
def print_board(board):
    display(SVG(chess.svg.board(board=board)))

# printing the legal moves
def print_legal_moves(board):
    print("Legal moves:")
    for move in board.legal_moves:
        print(move, end=" ")
    print()
    
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
    
