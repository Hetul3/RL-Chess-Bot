import torch
import torch.nn as nn
import torch.nn.functional as F
import chess

# most state of the art chess bots use CNNs in a residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ChessModel(nn.Module):
    def __init__(self, num_channels=14, num_filters=256, dropout_rate=0.3):
        super(ChessModel, self).__init__()
        # input channels: 14, num_filters: 256 number of feature maps that will be produced through dot producting the filter with the input, kernel_size: 3 size of the filter (3x3), padding: 1 will add 1 pizel to all sides of the input
        self.conv1 = nn.Conv2d(num_channels, num_filters, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_filters)
        
        # AlphaZero in the original paper had 19-39 residual blocks, we will use 10
        self.residual_block1 = ResidualBlock(num_filters, num_filters)
        self.residual_block2 = ResidualBlock(num_filters, num_filters)
        self.residual_block3 = ResidualBlock(num_filters, num_filters)
        self.residual_block4 = ResidualBlock(num_filters, num_filters)
        self.residual_block5 = ResidualBlock(num_filters, num_filters)
        self.residual_block6 = ResidualBlock(num_filters, num_filters)
        self.residual_block7 = ResidualBlock(num_filters, num_filters)
        self.residual_block8 = ResidualBlock(num_filters, num_filters)
        self.residual_block9 = ResidualBlock(num_filters, num_filters)
        self.residual_block10 = ResidualBlock(num_filters, num_filters)
        
        # flattens feature map to a 1D tensor and outputs 1024 features (converting to 1024 useful features about the board)
        self.fc1 = nn.Linear(num_filters * 8 * 8, 1024)
        # converting the features into total of 4096 outputs (all combination of a position to another position)
        self.fc2 = nn.Linear(1024, 64 * 64)
        
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.residual_block1(x)
        x = self.dropout(x)
        x = self.residual_block2(x)
        x = self.dropout(x)
        x = self.residual_block3(x)
        x = self.dropout(x)
        x = self.residual_block4(x)
        x = self.dropout(x)
        x = self.residual_block5(x)
        x = self.dropout(x)
        x = self.residual_block6(x)
        x = self.dropout(x)
        x = self.residual_block7(x)
        x = self.dropout(x)
        x = self.residual_block8(x)
        x = self.dropout(x)
        x = self.residual_block9(x)
        x = self.dropout(x)
        x = self.residual_block10(x)
        x = self.dropout(x)
        
        # reshape the tensor, flattening it from a (batch_size, 256, 8, 8), where the feature map is flattened so now it's a 2D tensor (batch_size, 256*8*8), -1 means the size of the tensor will be inferred
        x = x.view(-1, 256*8*8)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_move(self, board_tensor, legal_moves):
        with torch.no_grad():
            q_values = self(board_tensor.unsqueeze(0)).squeeze()
            
        best_move = None
        best_value = float('-inf')
        
        for move in legal_moves:
            from_square = move.from_square
            to_square = move.to_square
            move_value = q_values[from_square * 64 + to_square].item()
            if move_value > best_value:
                best_value = move_value
                best_move = move
            
        return best_move
    

def board_to_tensor(board):
    pieces = ['p', 'n', 'b', 'r', 'q', 'k', 'P', 'N', 'B', 'R', 'Q', 'K']
    # 14 channel tensor: first 12 channels represents all 12 pieces with a board for each piece showing where they are
    tensor = torch.zeros(14, 8, 8)
    for i, piece in enumerate(pieces):
        for square in board.pieces(chess.PIECE_SYMBOLS.index(piece.lower()), piece.isupper()):
            rank, file = divmod(square, 8)
            tensor[i][rank][file] = 1
            
    legal_moves = list(board.legal_moves)
    
    # 13th channel represents all the to_squares of the legal moves, all the options the model can go to
    for move in legal_moves:
        to_square = move.to_square
        rank, file = divmod(to_square, 8)
        tensor[12][rank][file] = 1
        
    # 14th channel represents all the from_squares of the legal moves, all the location of the pieces that can perform those legal moves
    for move in legal_moves:
        from_square = move.from_square
        rank, file = divmod(from_square, 8)
        tensor[13][rank][file] = 1
            
    return tensor

# Add a learning rate scheduler
def get_lr_scheduler(optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)