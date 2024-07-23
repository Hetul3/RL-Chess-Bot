from chess_env import ChessEnv
from model import ChessModel
from dqn_agent import DQNAgent
import torch
torch.set_num_threads(1)    # limiting threads running so it doesn't fry my macbook air :(

def train(episodes=1000, batch_size=128, target_update=10):
    env = ChessEnv(stockfish_path="stockfish/stockfish", skill_level=0, max_depth=3, blunder_probability=0.3, think_time=0.001)
    model = ChessModel()
    agent = DQNAgent(model)
    
    total_moves = 0
    total_games = 0
    total_loss = 0
    
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        evaluation_reward = 0
        done = False
        moves_in_game = 0
        
        while not done:
            legal_moves = env.get_legal_moves()
            action = agent.select_action(state, legal_moves)
            next_state, reward, done, _ = env.step(action)
            total_reward += reward
            evaluation_reward += reward - 1
            moves_in_game += 1
            
            agent.memory.push(state, action, next_state, reward)
            state = next_state
            
            loss = agent.train(batch_size)
            if loss is not None:
                total_loss += loss
            
        agent.update_epsilon()
        
        if episode % target_update == 0:
            agent.update_target_network()
        
        total_moves += moves_in_game
        total_games += 1
        avg_moves_per_game = total_moves / total_games
        avg_loss = total_loss / total_moves if total_moves > 0 else 0
        
        print(f"Episode {episode + 1}, Total Reward: {total_reward:.2f}, Evaluation Reward: {evaluation_reward:.2f}, "
              f"Epsilon: {agent.epsilon:.4f}, Moves: {moves_in_game}, Avg Moves: {avg_moves_per_game:.2f}, "
              f"Avg Loss: {avg_loss:.4f}")
        
    env.close()
    torch.save(model.state_dict(), "chess_model.pth")
    
if __name__ == "__main__":
    train()