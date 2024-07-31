from chess_env import ChessEnv
from model import ChessModel
from dqn_agent import DQNAgent
import torch
import os
import chess
torch.set_num_threads(3)    # limiting threads running so it doesn't fry my macbook air :(


def train(episodes=1000, batch_size=128, target_update=10, save_interval=5):
    env = ChessEnv(stockfish_path="stockfish/stockfish", skill_level=0, max_depth=None, blunder_probability=0.05, think_time=0.001)
    model = ChessModel()

    checkpoint_path = 'reinforced_checkpoint.pth'
    start_episode = 0
    
    if os.path.exists(checkpoint_path):
        print(f"Loading reinforcement checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_episode = checkpoint['episode']
        print(f"Resuming from checkpoint, Episode: {start_episode}")
    else:
        # If no reinforcement checkpoint, try loading the supervised model
        supervised_model_path = "supervised_model.pth"
        if os.path.exists(supervised_model_path):
            print(f"Loading pre-trained supervised model from {supervised_model_path}")
            model.load_state_dict(torch.load(supervised_model_path))
        else:
            print("No pre-trained model or checkpoint found, starting from scratch")

    agent = DQNAgent(model)
    
    if os.path.exists(checkpoint_path):
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    total_moves = 0
    total_games = 0
    total_loss = 0
    
    for episode in range(start_episode, start_episode + episodes):
        # reset everything for new game
        state = env.reset()
        total_reward = 0
        evaluation_reward = 0
        done = False
        moves_in_game = 0
        
        while not done:
            legal_moves = env.get_legal_moves()
            action = agent.select_action(state, legal_moves)    # bot selects move
            next_state, reward, done, _ = env.step(action)  # bot move is evaluated and stockfish makes a move
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
        
        if (episode + 1) % save_interval == 0:
            torch.save({
                'episode': episode+1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved at episode {episode + 1}")
    
    env.close()
    torch.save(model.state_dict(), "reinforced_model.pth")
    
if __name__ == "__main__":
    train()