"""
Base GNN class that all GNN models should inherit from.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class BaseGNN(nn.Module, ABC):
    """
    Abstract base class for all GNN models.
    
    This class provides the common interface and initialization that all GNN models
    should implement.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the base GNN model.
        
        Args:
            config: Configuration dictionary containing model parameters
        """
        super(BaseGNN, self).__init__()
        self.config = config
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Model parameters
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.5)
        
        # Handle hidden dimensions - can be single value or list
        hidden_dim = config['hidden_dim']
        if isinstance(hidden_dim, (list, tuple)):
            self.hidden_dims = list(hidden_dim)
            self.hidden_dim = hidden_dim[0]  # For backward compatibility
        else:
            self.hidden_dim = hidden_dim
            self.hidden_dims = [hidden_dim] * max(1, self.num_layers - 1)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 0.01)
        self.weight_decay = config.get('weight_decay', 5e-4)
        self.epochs = config.get('epochs', 200)
        self.optimizer_type = config.get('optimizer', 'adam').lower()
        
        # Optimizer-specific parameters
        self.momentum = config.get('momentum', 0.9)  # For SGD
        self.alpha = config.get('alpha', 0.99)  # For RMSprop
        self.eps = config.get('eps', 1e-8)  # For Adam, AdamW, RMSprop
        self.betas = config.get('betas', (0.9, 0.999))  # For Adam, AdamW
        
        # Initialize optimizer and loss function
        self.optimizer = None
        self.criterion = None
        
        # Move model to device
        self.to(self.device)
    
    def _init_optimizer_and_criterion(self):
        """Initialize optimizer and loss criterion."""
        if self.optimizer is None:
            self.optimizer = self._create_optimizer()
        
        if self.criterion is None:
            if self.config.get('task_type', 'classification') == 'classification':
                self.criterion = nn.CrossEntropyLoss()
            else:
                self.criterion = nn.MSELoss()
    
    def _create_optimizer(self):
        """Create optimizer based on the specified type."""
        params = self.parameters()
        
        if self.optimizer_type == 'adam':
            return optim.Adam(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=self.eps
            )
        elif self.optimizer_type == 'sgd':
            return optim.SGD(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                momentum=self.momentum
            )
        elif self.optimizer_type == 'rmsprop':
            return optim.RMSprop(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                alpha=self.alpha,
                eps=self.eps
            )
        elif self.optimizer_type == 'adamw':
            return optim.AdamW(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                betas=self.betas,
                eps=self.eps
            )
        elif self.optimizer_type == 'adagrad':
            return optim.Adagrad(
                params,
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
                eps=self.eps
            )
        else:
            raise ValueError(f"Unsupported optimizer type: {self.optimizer_type}. "
                           f"Supported types: adam, sgd, rmsprop, adamw, adagrad")
    
    def set_optimizer(self, optimizer_type: str, **optimizer_kwargs):
        """
        Change the optimizer type and parameters.
        
        Args:
            optimizer_type: Type of optimizer ('adam', 'sgd', 'rmsprop', 'adamw', 'adagrad')
            **optimizer_kwargs: Additional optimizer parameters to override defaults
        """
        self.optimizer_type = optimizer_type.lower()
        
        # Update optimizer parameters if provided
        if 'learning_rate' in optimizer_kwargs:
            self.learning_rate = optimizer_kwargs['learning_rate']
        if 'weight_decay' in optimizer_kwargs:
            self.weight_decay = optimizer_kwargs['weight_decay']
        if 'momentum' in optimizer_kwargs:
            self.momentum = optimizer_kwargs['momentum']
        if 'alpha' in optimizer_kwargs:
            self.alpha = optimizer_kwargs['alpha']
        if 'eps' in optimizer_kwargs:
            self.eps = optimizer_kwargs['eps']
        if 'betas' in optimizer_kwargs:
            self.betas = optimizer_kwargs['betas']
        
        # Recreate optimizer with new parameters
        self.optimizer = self._create_optimizer()
        print(f"Optimizer changed to {self.optimizer_type}")
    
    def get_optimizer_info(self) -> Dict[str, Any]:
        """Get information about the current optimizer."""
        info = {
            'type': self.optimizer_type,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay
        }
        
        if self.optimizer_type == 'sgd':
            info['momentum'] = self.momentum
        elif self.optimizer_type == 'rmsprop':
            info['alpha'] = self.alpha
            info['eps'] = self.eps
        elif self.optimizer_type in ['adam', 'adamw']:
            info['betas'] = self.betas
            info['eps'] = self.eps
        elif self.optimizer_type == 'adagrad':
            info['eps'] = self.eps
            
        return info
    
    @abstractmethod
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Node features tensor of shape [num_nodes, input_dim]
            edge_index: Edge indices tensor of shape [2, num_edges]
            **kwargs: Additional arguments specific to the model
            
        Returns:
            Output tensor of shape [num_nodes, output_dim]
        """
        pass
    
    def train_step(self, data, mask=None) -> float:
        """
        Perform a single training step.
        
        Args:
            data: Training data with x (node features), edge_index (edges), and y (labels)
            mask: Optional mask for selecting specific nodes for training
            
        Returns:
            Loss value for this training step
        """
        # Initialize optimizer and criterion if not already done
        self._init_optimizer_and_criterion()
        
        self.train()
        self.optimizer.zero_grad()
        
        # Move data to device
        data = self._to_device(data)
        
        # Forward pass
        out = self.forward(data.x, data.edge_index)
        
        # Compute loss
        if mask is not None:
            mask = mask.to(self.device)
            loss = self.criterion(out[mask], data.y[mask])
        else:
            loss = self.criterion(out, data.y)
        
        # Backward pass
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def eval_step(self, data, mask=None) -> Dict[str, float]:
        """
        Perform evaluation step.
        
        Args:
            data: Evaluation data
            mask: Optional mask for selecting specific nodes for evaluation
            
        Returns:
            Dictionary containing loss and accuracy
        """
        self._init_optimizer_and_criterion()
        self.eval()
        
        with torch.no_grad():
            # Move data to device
            data = self._to_device(data)
            
            # Forward pass
            out = self.forward(data.x, data.edge_index)
            
            # Compute loss and accuracy
            if mask is not None:
                mask = mask.to(self.device)
                loss = self.criterion(out[mask], data.y[mask])
                pred = out[mask].argmax(dim=1)
                correct = pred.eq(data.y[mask]).sum().item()
                accuracy = correct / mask.sum().item()
            else:
                loss = self.criterion(out, data.y)
                pred = out.argmax(dim=1)
                correct = pred.eq(data.y).sum().item()
                accuracy = correct / data.y.size(0)
                
        return {'loss': loss.item(), 'accuracy': accuracy}
    
    def predict(self, data) -> torch.Tensor:
        """
        Make predictions on new data.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Predictions tensor
        """
        self.eval()
        with torch.no_grad():
            data = self._to_device(data)
            return self.forward(data.x, data.edge_index)
    
    def _to_device(self, data):
        """Move data to the model's device."""
        if hasattr(data, 'to'):
            return data.to(self.device)
        return data
    
    def save_model(self, path: str):
        """Save the model state."""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'config': self.config,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'criterion_state_dict': self.criterion.state_dict() if self.criterion else None
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load the model state."""
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint['model_state_dict'])
        
        if self.optimizer and checkpoint.get('optimizer_state_dict'):
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.criterion and checkpoint.get('criterion_state_dict'):
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            
        print(f"Model loaded from {path}")
    
    def get_num_parameters(self) -> int:
        """Get the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def reset_parameters(self):
        """Reset all parameters to their initial values."""
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive information about the model."""
        info = {
            'model_name': self.__class__.__name__,
            'input_dim': self.input_dim,
            'hidden_dim': self.hidden_dim,
            'hidden_dims': self.hidden_dims,
            'output_dim': self.output_dim,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'num_parameters': self.get_num_parameters(),
            'device': str(self.device),
            'optimizer': self.get_optimizer_info()
        }
        return info 
    
    