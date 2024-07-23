import chess
import chess.engine
import random
from model import board_to_tensor

class ChessEnv:
    def __init__(self, stockfish_path, skill_level=0, max_depth=1, blunder_probability=1, think_time=0.03):
        self.board = chess.Board()
        self.stockfish = chess.engine.SimpleEngine.popen_uci(stockfish_path)
        self.previous_evaluation = 0
        self.set_stockfish_skill_level(skill_level)
        self.max_depth = max_depth
        self.blunder_probability = blunder_probability
        self.think_time = think_time
        
    def set_stockfish_skill_level(self, skill_level):
        if 0 <= skill_level <= 20:
            self.stockfish.configure({"Skill Level": skill_level})
        else:
            raise ValueError("Skill level must be between 0 and 20")
        
    def reset(self):
        self.board.reset()
        self.previous_evaluation = 0
        return board_to_tensor(self.board)
    
    def get_board_evaluation(self):
        try:
            result = self.stockfish.analyse(self.board, chess.engine.Limit(time=self.think_time, depth=self.max_depth))
            score = result['score'].white().score()
            print(f"Score: {score}")
            if score is None:
                if self.previous_evaluation is not None:
                    return self.previous_evaluation
                else:
                    return 0
            return score
        except chess.engine.EngineTerminatedError as e:
            print(f"Stockfish engine has terminated: {e}")
        except Exception as e:
            print(f"Error in get_board_evaluation: {e}")
            if self.previous_evaluation is not None:
                return self.previous_evaluation
            else:
                return 0
    
    def calculate_evaluation_reward(self, current_evaluation):
        if current_evaluation is None:
            reward = 0
        else:
            reward = (current_evaluation - self.previous_evaluation)
        self.previous_evaluation = current_evaluation if current_evaluation is not None else self.previous_evaluation
        return reward
    
    def step(self, move):
        if move in self.board.legal_moves:
            self.board.push(move)
        else:
            return board_to_tensor(self.board), -10, True, "Illegal move"
        
        current_evaluation = self.get_board_evaluation()
        evaluation_reward = float(self.calculate_evaluation_reward(current_evaluation)) / 100.0
        
        if self.board.is_game_over():
            print(self.board)
            if self.board.result() == "1-0":
                print("Win")
                return board_to_tensor(self.board), 100 + evaluation_reward, True, "Win"
            elif self.board.result() == "0-1":
                print("Loss")
                return board_to_tensor(self.board), -5 + evaluation_reward, True, "Loss"
            else:
                print("Draw")
                return board_to_tensor(self.board), 10 + evaluation_reward, True, "Draw"
        
        result = self.stockfish.play(self.board, chess.engine.Limit(time=self.think_time, depth=self.max_depth))
        
        if random.random() < self.blunder_probability:
            legal_moves = list(self.board.legal_moves)
            if legal_moves:
                result.move = random.choice(legal_moves)
        
        self.board.push(result.move)
        
        if not self.board.is_game_over():
            current_evaluation = self.get_board_evaluation()
            evaluation_reward += float(self.calculate_evaluation_reward(current_evaluation)) / 100.0
        
        if self.board.is_game_over():
            print(self.board)
            if self.board.result() == "1-0":
                print("Win")
                return board_to_tensor(self.board), 100 + evaluation_reward, True, "Win"
            elif self.board.result() == "0-1":
                print("Loss")
                return board_to_tensor(self.board), -5 + evaluation_reward, True, "Loss"
            else:
                print("Draw")
                return board_to_tensor(self.board), 10 + evaluation_reward, True, "Draw"
            
        return board_to_tensor(self.board), 0.5 + evaluation_reward, False, "Ongoing"
        
    def close(self):
        self.stockfish.quit()
        
    def get_legal_moves(self):
        return list(self.board.legal_moves)