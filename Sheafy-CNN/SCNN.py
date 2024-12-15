# Author Vicente Gonzalez Bosca (vicenteg@sas.upenn.edu)
# Last updated 14th December 2024
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageDraw
import math

# image visualization

def show_images(loader):
    """Display a grid of 25 images from the dataset"""
    # Get a batch of images
    examples = iter(loader)
    images, labels = next(examples)
    
    # Create a 5x5 grid
    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i][0], cmap='gray')
        plt.title(f'Label: {labels[i].item()}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def show_single_image(dataset, index=0):
    # Get a single image and its label
    image, label = dataset[index]
    
    # Convert the image from tensor [1, 28, 28] to numpy array [28, 28]
    # We need to squeeze out the channel dimension and detach from computation graph
    image_array = image.squeeze().detach().numpy()
    
    # Create a figure
    plt.figure(figsize=(6, 6))
    
    # Display the image
    plt.imshow(image_array, cmap='gray')
    plt.title(f'Label: {label}')
    plt.axis('off')
    
    # Add colorbar to show pixel value scale
    plt.colorbar()
    
    # Print the actual pixel values
    print("Image shape:", image_array.shape)
    print("\nPixel value range:")
    print(f"Min value: {image_array.min():.3f}")
    print(f"Max value: {image_array.max():.3f}")
    print(f"Mean value: {image_array.mean():.3f}")
    
    plt.show()

# CNN class
class MNISTConvNet(nn.Module):
    def __init__(
        self,
        in_channels=1,
        conv_channels=[32, 64],  # List of output channels for each conv layer
        conv_kernel_size=5,
        pool_size=2,
        pool_stride=2,
        fc_features=[512],  # List of features for each fc layer (excluding output)
        dropout_rate=0.25,
        num_classes=10,
        input_size=28,  # Size of input image (assuming square)
        activation=F.relu
    ):
        super().__init__()

        # Save configuration
        self.config = {
            'in_channels': in_channels,
            'conv_channels': conv_channels,
            'conv_kernel_size': conv_kernel_size,
            'pool_size': pool_size,
            'pool_stride': pool_stride,
            'fc_features': fc_features,
            'dropout_rate': dropout_rate,
            'num_classes': num_classes,
            'input_size': input_size,
            'activation': activation
        }
        
        # Save hyperparameters
        self.conv_kernel_size = conv_kernel_size
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dropout_rate = dropout_rate
        
        # Create convolutional layers
        self.conv_layers = nn.ModuleList()
        current_channels = in_channels
        current_size = input_size

        # Save activation function
        self.activation = activation
        
        for out_channels in conv_channels:
            self.conv_layers.append(
                nn.Conv2d(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=conv_kernel_size
                )
            )
            # Update size after convolution
            current_size = current_size - conv_kernel_size + 1
            # Update size after pooling
            current_size = current_size // pool_size
            current_channels = out_channels
        
        # Calculate size of flattened features after last conv layer
        self.flatten_size = current_channels * current_size * current_size
        
        # Create fully connected layers
        self.fc_layers = nn.ModuleList()
        current_features = self.flatten_size
        
        for fc_size in fc_features:
            self.fc_layers.append(nn.Linear(current_features, fc_size))
            current_features = fc_size
        
        # Output layer
        self.fc_out = nn.Linear(current_features, num_classes)
        
        # Pooling and dropout
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Convolutional layers
        for conv in self.conv_layers:
            x = conv(x)
            x = self.activation(x)
            x = self.pool(x)
        
        # Flatten
        x = x.view(-1, self.flatten_size)
        
        # Fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Output layer
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)
    
    def print_model_info(self, name):
        print(f"\n{name}:")
        print(self)
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def save(self, path):
        """Save the model weights and architecture configuration"""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
        
    @classmethod
    def load(cls, path):
        """Load a model with its weights and configuration"""
        checkpoint = torch.load(path)
        
        # Create a new model with the saved configuration
        model = cls(**checkpoint['config'])
        
        # Load the weights
        model.load_state_dict(checkpoint['state_dict'])
        
        return model
    
    def save_weights(self, path):
        """Save only the model weights"""
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """Load only the model weights"""
        self.load_state_dict(torch.load(path))

class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        test_loader,
        optimizer,
        criterion=nn.NLLLoss(reduction='mean'),
        device='cpu',
        patience=5,  # Number of epochs to wait before early stopping
        min_delta=0.001  # Minimum change in validation loss to qualify as an improvement
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        self.min_delta = min_delta
        
        # Move model to device
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.test_accuracies = []
        self.best_accuracy = 0
        self.best_val_loss = float('inf')
        self.current_epoch = 0
        
        # Early stopping attributes
        self.counter = 0
        self.early_stop = False
        self.best_model_state = None
        
    def train_epoch(self):
        """Run one epoch of training"""
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {self.current_epoch+1} '
                      f'[{batch_idx * len(data)}/{len(self.train_loader.dataset)} '
                      f'({100. * batch_idx / len(self.train_loader):.0f}%)]'
                      f'\tLoss: {loss.item():.6f}')
                
        return running_loss / len(self.train_loader)
    
    def evaluate(self, loader, purpose='validation'):
        """Evaluate the model on the given loader"""
        self.model.eval()
        running_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                running_loss += self.criterion(output, target).item() * len(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += len(data)
        
        avg_loss = running_loss / total
        accuracy = 100. * correct / total
        
        return avg_loss, accuracy
    
    def check_early_stopping(self, val_loss):
        """Check if training should be stopped"""
        if val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_model_state = self.model.state_dict().copy()
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
            return False
    
    def train(self, epochs, save_path='best_model.pth'):
        """Train the model with early stopping"""
        import time
        start_time = time.time()

        # Create paths for model and training state
        model_path = save_path
        training_path = save_path.replace('.pth', '_training.pth')
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Training
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validation
            val_loss, val_accuracy = self.evaluate(self.val_loader, 'validation')
            self.val_losses.append(val_loss)
            
            # Test
            test_loss, test_accuracy = self.evaluate(self.test_loader, 'test')
            self.test_accuracies.append(test_accuracy)
            
            # Print epoch summary
            epoch_time = time.time() - epoch_start
            print(f'\nEpoch {epoch+1} Complete:')
            print(f'Training Loss: {train_loss:.4f}')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Test Accuracy: {test_accuracy:.2f}%')
            print(f'Time taken: {epoch_time:.2f} seconds')
            
            # Check early stopping
            if self.check_early_stopping(val_loss):
                print(f'\nEarly stopping triggered after epoch {epoch+1}')
                # Restore best model
                self.model.load_state_dict(self.best_model_state)
                break
            
            # Save if best accuracy
            if test_accuracy > self.best_accuracy:
                self.best_accuracy = test_accuracy
                # Save model using model's save method
                self.model.save(model_path)
                # Save training state separately
                self.save_training_state(training_path)
        
        total_time = time.time() - start_time
        print(f'\nTraining complete in {total_time/60:.2f} minutes')
        print(f'Best test accuracy: {self.best_accuracy:.2f}%')
        
        return self.train_losses, self.val_losses, self.test_accuracies
    
    def save_training_state(self, path):
        """Save only training-related state"""
        torch.save({
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_accuracies': self.test_accuracies,
            'current_epoch': self.current_epoch,
            'best_accuracy': self.best_accuracy,
            'best_val_loss': self.best_val_loss
        }, path)

    def load_training_state(self, path):
        """Load only training-related state"""
        checkpoint = torch.load(path)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.test_accuracies = checkpoint['test_accuracies']
        self.current_epoch = checkpoint['current_epoch']
        self.best_accuracy = checkpoint['best_accuracy']
        self.best_val_loss = checkpoint['best_val_loss']

    def plot_training_curves(self):
        """Plot training curves including losses and accuracy"""
        plt.figure(figsize=(15, 5))
        
        # Loss curves
        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        
        # Accuracy curve
        plt.subplot(1, 2, 2)
        plt.plot(self.test_accuracies, label='Test Accuracy', color='green')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Test Accuracy over Time')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()

    def analyze_predictions(self, num_samples=10):
        """Analyze model predictions including confusion matrix and sample predictions"""
        self.model.eval()
        all_preds = []
        all_targets = []
        sample_images = []
        sample_preds = []
        sample_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                
                # Store predictions and targets for confusion matrix
                all_preds.extend(pred.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
                
                # Store some samples for visualization
                if len(sample_images) < num_samples:
                    batch_samples = min(num_samples - len(sample_images), len(data))
                    sample_images.extend(data.cpu()[:batch_samples])
                    sample_preds.extend(pred.cpu()[:batch_samples])
                    sample_targets.extend(target.cpu()[:batch_samples])
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(all_targets, all_preds)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        # Plot sample predictions
        plt.figure(figsize=(20, 4))
        for i in range(num_samples):
            plt.subplot(2, 5, i + 1)
            plt.imshow(sample_images[i].squeeze(), cmap='gray')
            color = 'green' if sample_preds[i] == sample_targets[i] else 'red'
            plt.title(f'Pred: {sample_preds[i].item()}\nTrue: {sample_targets[i].item()}', 
                     color=color)
            plt.axis('off')
        plt.show()
        
        # Calculate per-class metrics
        print("\nPer-class Analysis:")
        print(classification_report(all_targets, all_preds))
        
        return all_preds, all_targets

    def analyze_performance(self):
        """Comprehensive model performance analysis"""
        print("Model Performance Analysis")
        print("=" * 50)
        
        # 1. Plot training curves
        self.plot_training_curves()
        
        # 2. Print final metrics
        print("\nFinal Metrics:")
        print(f"Best Test Accuracy: {self.best_accuracy:.2f}%")
        print(f"Final Training Loss: {self.train_losses[-1]:.4f}")
        print(f"Final Validation Loss: {self.val_losses[-1]:.4f}")
        
        # 3. Learning convergence
        print("\nLearning Convergence Analysis:")
        n_epochs = len(self.train_losses)
        print(f"Total Epochs Trained: {n_epochs}")
        print(f"Epoch with Best Performance: {np.argmax(self.test_accuracies) + 1}")
        
        # Calculate if training was still improving
        last_epochs = 5
        if n_epochs >= last_epochs:
            recent_improvement = (np.mean(self.val_losses[-last_epochs:]) - 
                                self.val_losses[-1])
            print(f"Recent Improvement (last {last_epochs} epochs): {recent_improvement:.6f}")
        
        # 4. Analyze predictions
        print("\nDetailed Prediction Analysis:")
        self.analyze_predictions()
        
        # 5. Model Complexity
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nModel Complexity:")
        print(f"Total Parameters: {total_params:,}")
        print(f"Trainable Parameters: {trainable_params:,}")

def create_trainer(
    model, 
    train_dataset, 
    test_dataset,  # Added test_dataset as required parameter
    val_size=0.1, 
    batch_size=64, 
    learning_rate=0.001, 
    patience=5
):
    """Create a trainer with validation split
    
    Args:
        model: The neural network model
        train_dataset: The training dataset
        test_dataset: The test dataset
        val_size: Fraction of training data to use for validation
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        patience: Number of epochs to wait before early stopping
    """
    # Calculate split sizes
    val_length = int(len(train_dataset) * val_size)
    train_length = len(train_dataset) - val_length
    
    # Split dataset
    train_subset, val_subset = torch.utils.data.random_split(
        train_dataset, 
        [train_length, val_length]
    )
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        patience=patience
    )
    
    return trainer

# Draw a number widget
class DigitDrawer:
    def __init__(self, model):
        # Load the trained model
        self.model = model  # Make sure this matches your trained architecture
        self.model.eval()
        
        # Setup preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        
        # Create main window
        self.root = tk.Tk()
        self.root.title("Draw a Digit")
        
        # Drawing area size
        self.canvas_size = 280  # Larger for better drawing
        self.drawing_width = 20  # Width of the drawing line
        
        # Create canvas
        self.canvas = tk.Canvas(
            self.root, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg='black'
        )
        self.canvas.grid(row=0, column=0, columnspan=2, padx=5, pady=5)
        
        # Create buttons
        self.clear_button = ttk.Button(
            self.root, 
            text="Clear", 
            command=self.clear_canvas
        )
        self.clear_button.grid(row=1, column=0, padx=5, pady=5)
        
        self.predict_button = ttk.Button(
            self.root, 
            text="Predict", 
            command=self.predict_digit
        )
        self.predict_button.grid(row=1, column=1, padx=5, pady=5)
        
        # Create prediction label
        self.prediction_label = ttk.Label(
            self.root, 
            text="Draw a digit and click Predict",
            font=('Arial', 14)
        )
        self.prediction_label.grid(row=2, column=0, columnspan=2, pady=5)
        
        # Drawing state
        self.last_x = None
        self.last_y = None
        
        # Bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)
        
        # Create image buffer
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
    
    def start_drawing(self, event):
        self.last_x = event.x
        self.last_y = event.y
    
    def draw(self, event):
        if self.last_x and self.last_y:
            self.canvas.create_line(
                self.last_x, self.last_y, 
                event.x, event.y, 
                width=self.drawing_width, 
                fill='white', 
                capstyle=tk.ROUND, 
                smooth=True
            )
            self.draw.line(
                [self.last_x, self.last_y, event.x, event.y], 
                fill='white', 
                width=self.drawing_width
            )
        self.last_x = event.x
        self.last_y = event.y
    
    def stop_drawing(self, event):
        self.last_x = None
        self.last_y = None
    
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (self.canvas_size, self.canvas_size), 'black')
        self.draw = ImageDraw.Draw(self.image)
        self.prediction_label.config(text="Draw a digit and click Predict")
    
    def predict_digit(self):
        # Convert drawing to MNIST format
        # First, resize to 28x28
        img_array = np.array(self.image.resize((28, 28)))
        
        # Convert to tensor and normalize
        img_tensor = self.transform(Image.fromarray(img_array)).unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            output = self.model(img_tensor)
            pred = output.argmax(dim=1, keepdim=True)
            prob = F.softmax(output, dim=1)
            confidence = prob[0][pred].item() * 100
        
        # Update label with prediction and confidence
        self.prediction_label.config(
            text=f"Prediction: {pred.item()} (Confidence: {confidence:.2f}%)"
        )
    
    def run(self):
        self.root.mainloop()


### Sheafy CNN building blocks ###

class NodestoEdges(nn.Module):
    """Transforms node (pixel) information to edge information using a sheaf-based approach.
    
    Args:
        m (int): Number of rows in the pixel grid
        n (int): Number of columns in the pixel grid
        in_channels (int): Number of input features per pixel
        out_channels (int): Number of output features per edge
        mix_channels (bool): If True, allows information flow between different channels.
                           If False, each input channel only affects its corresponding output channel.
                           When False, in_channels must equal out_channels.
    
    The transformation creates a sparse matrix that connects pixels to their boundary edges.
    Each edge connects two adjacent pixels, either horizontally or vertically.
    """
    def __init__(self, m, n, in_channels, out_channels, mix_channels=True, bias=True):
        
        # Initialize parent
        super().__init__()
        
        # Save filter hyperparameters
        self.m = m                 # number of rows
        self.n = n                 # number of columns
        self.k = m*(n-1) + n*(m-1) # number of edges: horizontal edges + vertical edges
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mix_channels = mix_channels
        # Make bias optional
        self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None

        if not mix_channels:
            # Ensure channels match when not mixing
            # Only enforce channel matching if in_channels > 1
            if in_channels > 1:
                assert in_channels == out_channels, "Channels must match when mix_channels=False"
            # Create weight tensor [channels, edges*2] - each channel processed independently
            self.weight = nn.Parameter(torch.randn(in_channels, 2*self.k))
        else:
            # Create weight tensor [out_channels, in_channels, edges*2] for full mixing
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 2*self.k))
        
        self.reset_parameters()
        self.create_delta_sparse()
        
    def reset_parameters(self):
        # Initialize weights uniformly based on tensor size
        stdv = 1. / math.sqrt(2*self.k*self.in_channels)
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def create_delta_sparse(self):
        indices = [] # indices of non-zero entries
        # Create horizontal edge connections
        for i in range(self.m):
            for j in range(self.n-1):
                # Connect edge to its left pixel
                indices.append((i*(self.n-1) + j, i*(self.n-1) + i + j))
                # Connect edge to its right pixel
                indices.append((i*(self.n-1) + j, i*(self.n-1) + i + j + 1))
        
        # Create vertical edge connections
        indices += list(zip(range(self.m*(self.n-1), self.m*(self.n-1) +  self.n*(self.m - 1)), range(self.n*(self.m - 1))))
        indices += list(zip(range(self.m*(self.n-1), self.m*(self.n-1) +  self.n*(self.m - 1)), range(self.n, self.n + self.n*(self.m - 1))))
        
        # Convert indices to proper format for sparse tensor
        indices = torch.tensor(indices).t()
        
        # Create sparse tensor structure with dummy ones
        self.delta = torch.sparse_coo_tensor(
        indices, 
        torch.ones(2*self.k), # placeholder values for structure
        size=(self.k, self.n*self.m)  # [edges, pixels]
    )

    def forward(self, x):
        # x shape: [batch_size, in_channels, m*n]
        batch_size = x.size(0)

        # Get optimized sparse tensor structure
        base_delta = self.delta.coalesce()
        indices = base_delta.indices()
        
        output = []
        if self.mix_channels:
            # Process with channel mixing
            for out_c in range(self.out_channels):
                channel_output = []
                for in_c in range(self.in_channels):
                    # Create sparse operator for this channel combination
                    channel_weights = self.weight[out_c, in_c]
                    channel_delta = torch.sparse_coo_tensor(
                        indices,
                        channel_weights,
                        size=self.delta.size()
                    )
                    # Apply operator to input channel
                    x_reshaped = x[:, in_c].reshape(batch_size, -1)
                    result = torch.sparse.mm(channel_delta, x_reshaped.t())
                    channel_output.append(result.t())
                # Sum all input channels for this output channel
                result = torch.sum(torch.stack(channel_output), dim=0)
                if self.bias is not None:
                    result = result + self.bias[out_c].view(1, -1)
                output.append(result)
        else: 
            # Process each channel independently
            for c in range(self.in_channels):
                channel_weights = self.weight[c]
                channel_delta = torch.sparse_coo_tensor(
                    indices,
                    channel_weights,
                    size=self.delta.size()
                )
                x_reshaped = x[:, c].reshape(batch_size, -1)
                result = torch.sparse.mm(channel_delta, x_reshaped.t())
                if self.bias is not None:
                    result = result.t() + self.bias[c].view(1, -1)
                    output.append(result)
                else:
                    output.append(result.t())
                    
        return torch.stack(output, dim=1)  # [batch_size, out_channels, k]
    

class EdgestoInter(nn.Module):
   """Transforms edge information to intersection information using a sheaf-based approach.
   
   Args:
       m (int): Number of rows in the pixel grid 
       n (int): Number of columns in the pixel grid
       in_channels (int): Number of input features per edge
       out_channels (int): Number of output features per intersection
       mix_channels (bool): If True, allows information flow between different channels.
                           If False, each input channel only affects its corresponding output channel.
                           When False, in_channels must equal out_channels.
       
   The transformation creates a sparse matrix that connects edges to their intersections.
   Each intersection point is connected to 4 edges - 2 horizontal and 2 vertical.
   Number of intersections is (m-1)*(n-1) since they occur at edge crossings.
   """
   def __init__(self, m, n, in_channels, out_channels, mix_channels=True, bias=True):
       
       # Initialize parent
       super().__init__()
       
       # Save filter hyperparameters
       self.m = m                 # number of rows
       self.n = n                 # number of columns
       self.k = (m-1)*(n-1)      # number of intersections (where edges cross)
       self.in_channels = in_channels
       self.out_channels = out_channels
       self.mix_channels = mix_channels
       # Make bias optional
       self.bias = nn.Parameter(torch.zeros(out_channels)) if bias else None
    
       if not mix_channels:
            # Ensure channels match when not mixing
            # Only enforce channel matching if in_channels > 1
            if in_channels > 1:
                assert in_channels == out_channels, "Channels must match when mix_channels=False"
            # Create weight tensor [channels, intersections*4] - each channel processed independently
            self.weight = nn.Parameter(torch.randn(in_channels, 4*self.k))
       else:
            # Create weight tensor [out_channels, in_channels, intersections*4] for full mixing
            self.weight = nn.Parameter(torch.randn(out_channels, in_channels, 4*self.k))
       self.reset_parameters()
       self.create_delta_sparse()
       
   def reset_parameters(self):
       # Initialize weights uniformly based on tensor size
       stdv = 1. / math.sqrt(4*self.k*self.in_channels)
       self.weight.data.uniform_(-stdv, stdv)
       if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

   def create_delta_sparse(self):
       indices = [] # indices of non-zero entries
       
       # Connect intersections to horizontal edges
       indices += list(zip(range((self.m-1)*(self.n-1)), range((self.m-1)*(self.n-1))))
       indices += list(zip(range((self.m-1)*(self.n-1)), range(self.n - 1, self.n - 1 + (self.m-1)*(self.n-1))))
       
       # Connect intersections to vertical edges
       for i in range(self.m-1):
           for j in range(self.n-1):
               indices.append((i*(self.n-1) + j, (self.n-1)*self.m + i*(self.n-1) + i + j))
               indices.append((i*(self.n-1) + j, (self.n-1)*self.m + i*(self.n-1) + i + j + 1))
       
       # Convert indices to proper format for sparse tensor
       indices = torch.tensor(indices).t()
       
       # Create sparse tensor structure with dummy ones
       self.delta = torch.sparse_coo_tensor(
       indices, 
       torch.ones(4*self.k), # placeholder values for structure
       size=(self.k, self.m*(self.n-1) + self.n*(self.m-1))  # [intersections, edges]
   )

   def forward(self, x):
        # x shape: [batch_size, in_channels, m*n]
        batch_size = x.size(0)

        # Get optimized sparse tensor structure
        base_delta = self.delta.coalesce()
        indices = base_delta.indices()
        
        output = []
        if self.mix_channels:
            # Process with channel mixing
            for out_c in range(self.out_channels):
                channel_output = []
                for in_c in range(self.in_channels):
                    # Create sparse operator for this channel combination
                    channel_weights = self.weight[out_c, in_c]
                    channel_delta = torch.sparse_coo_tensor(
                        indices,
                        channel_weights,
                        size=self.delta.size()
                    )
                    # Apply operator to input channel
                    x_reshaped = x[:, in_c].reshape(batch_size, -1)
                    result = torch.sparse.mm(channel_delta, x_reshaped.t())
                    channel_output.append(result.t())
                # Sum all input channels for this output channel
                result = torch.sum(torch.stack(channel_output), dim=0)
                if self.bias is not None:
                    result = result + self.bias[out_c].view(1, -1)
                output.append(result)
        else: 
            # Process each channel independently
            for c in range(self.in_channels):
                channel_weights = self.weight[c]
                channel_delta = torch.sparse_coo_tensor(
                    indices,
                    channel_weights,
                    size=self.delta.size()
                )
                x_reshaped = x[:, c].reshape(batch_size, -1)
                result = torch.sparse.mm(channel_delta, x_reshaped.t())
                if self.bias is not None:
                    result = result.t() + self.bias[c].view(1, -1)
                    output.append(result)
                else:
                    output.append(result.t())
                    
        return torch.stack(output, dim=1)  # [batch_size, out_channels, k]


# Concatenate both NodestoEdges and EdgestoInter in a block   
class SheafyConvBlock(nn.Module):
    """
    A neural network block inspired by cellular sheaves that transforms image data through
    a sequence of sheaf-theoretic operations.
    
    Parameters:
    -----------
    m : int
        Height of the input image
    n : int
        Width of the input image
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels
    bias : bool, default=True
        Whether to add a bias term to the final edges_to_inter operation
    mix_middle : bool, default=False
        If True, allows mixing between channels in the nodes_to_edges operation
    mix_end : bool, default=True
        If True, allows mixing between channels in the edges_to_inter operation
    inner_bias : bool, default=False
        Whether to add a bias term to the nodes_to_edges operation
    inner_activation : callable, optional
        Activation function to apply after the nodes_to_edges operation
    """
    def __init__(self, m, n, in_channels, out_channels, bias=True, 
                 mix_middle=False, mix_end=True, inner_bias=False, 
                 inner_activation=None):
        super().__init__()

        # Store spatial dimensions
        self.m = m
        self.n = n

        # Ensure channels match when no mixing is allowed
        if not mix_middle and not mix_end and in_channels != out_channels:
            raise AssertionError("When mix_middle=False and mix_end=False, in_channels must equal out_channels")

        # Determine number of channels after nodes_to_edges operation
        # For in_channels=1, mix_middle=True expands dimension
        # For in_channels>1, default behavior preserves channels
        if mix_middle:
            mid_channels = out_channels
        else:
            mid_channels = in_channels
        
        # First layer: transform node (pixel) data to edge data
        self.nodes_to_edges = NodestoEdges(
            m=m, n=n, 
            in_channels=in_channels, 
            out_channels=mid_channels,
            mix_channels=mix_middle,
            bias=inner_bias
        )

        # Second layer: transform edge data to intersection data
        # mix_end=False recommended when mix_middle=True to avoid double mixing
        self.edges_to_inter = EdgestoInter(
            m=m, n=n,
            in_channels=mid_channels,
            out_channels=out_channels,
            mix_channels=mix_end,
            bias=bias
        )
        
        # Store optional activation for after nodes_to_edges
        self.inner_activation = inner_activation
        
    
    def forward(self, x):
        """
        Forward pass through the sheaf block.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, channels, H, W]
            
        Returns:
        --------
        torch.Tensor
            Output tensor of shape [batch_size, out_channels, H-1, W-1]
        """
        # Verify input dimensions match expected size
        batch_size, channels, H, W = x.size()
        assert H == self.m and W == self.n, f"Input spatial dimensions {H}x{W} do not match expected {self.m}x{self.n}"

        # Reshape input for sheaf operations
        x = x.view(batch_size, channels, -1)

        # Apply sheaf transformations
        x = self.nodes_to_edges(x)
        
        # Apply optional activation between transformations
        if self.inner_activation is not None:
            x = self.inner_activation(x)
            
        x = self.edges_to_inter(x)
        
        # Reshape output back to 2D spatial format
        x = x.view(batch_size, -1, self.m-1, self.n-1)
        return x
    

# Standard sheafy net architecture for MNIST based on MNISTConvNet
class SheafyNet(nn.Module):
    """
    A complete neural network architecture using sheaf-inspired convolutional blocks
    for image classification tasks.
    
    Parameters:
    -----------
    in_channels : int, default=1
        Number of input channels in the image
    sheafy_channels : list of int, default=[32, 64]
        Number of output channels for each sheafy convolutional layer
    mix_middle : bool or list of bool, default=False
        Whether to allow channel mixing in nodes_to_edges operation for each layer
    mix_end : bool or list of bool, default=True
        Whether to allow channel mixing in edges_to_inter operation for each layer
    inner_bias : bool or list of bool, default=False
        Whether to use bias in nodes_to_edges operation for each layer
    inner_activation : callable, optional
        Activation function to use between sheaf operations
    pool_size : int, default=2
        Size of pooling window
    pool_stride : int, default=2
        Stride of pooling operation
    fc_features : list of int, default=[512]
        Number of features in each fully connected layer (excluding output)
    dropout_rate : float, default=0.25
        Dropout probability
    num_classes : int, default=10
        Number of output classes
    input_size : int, default=28
        Size of input image (assuming square)
    activation : callable, default=F.relu
        Activation function to use throughout the network
    """
    def __init__(
        self,
        in_channels=1,
        sheafy_channels=[32, 64],  
        mix_middle=False,  
        mix_end=True,     
        inner_bias=False, 
        inner_activation=None,
        pool_size=2,
        pool_stride=2,
        fc_features=[512],  
        dropout_rate=0.25,
        num_classes=10,
        input_size=28,  
        activation=F.relu
    ):
        super().__init__()

        # Store configuration for model saving/loading
        self.config = {
            'in_channels': in_channels,
            'sheafy_channels': sheafy_channels,
            'mix_middle': mix_middle,
            'mix_end': mix_end,
            'inner_bias': inner_bias,
            'inner_activation': inner_activation,
            'pool_size': pool_size,
            'pool_stride': pool_stride,
            'fc_features': fc_features,
            'dropout_rate': dropout_rate,
            'num_classes': num_classes,
            'input_size': input_size,
            'activation': activation
        }

        # Convert single boolean values to lists for each layer
        if isinstance(mix_middle, bool):
            mix_middle = [mix_middle] * len(sheafy_channels)
        if isinstance(mix_end, bool):
            mix_end = [mix_end] * len(sheafy_channels)
        if isinstance(inner_bias, bool):
            inner_bias = [inner_bias] * len(sheafy_channels)

        # Store hyperparameters as instance variables
        self.pool_size = pool_size
        self.pool_stride = pool_stride
        self.dropout_rate = dropout_rate
        self.activation = activation
        
        # Build sheafy convolutional layers
        self.sheafy_layers = nn.ModuleList()
        current_channels = in_channels
        current_size = input_size
        
        # Create each sheafy layer while tracking dimensions
        for i, out_channels in enumerate(sheafy_channels):
            self.sheafy_layers.append(
                SheafyConvBlock(
                    m=current_size,
                    n=current_size,
                    in_channels=current_channels,
                    out_channels=out_channels,
                    mix_middle=mix_middle[i],
                    mix_end=mix_end[i],
                    inner_bias=inner_bias[i],
                    inner_activation=inner_activation
                )
            )
            # Update dimensions after sheaf operation (reduces by 1)
            current_size = current_size - 1
            # Update dimensions after pooling
            current_size = current_size // pool_size
            current_channels = out_channels
        
        # Calculate flattened feature size for FC layers
        self.flatten_size = current_channels * current_size * current_size
        
        # Build fully connected layers
        self.fc_layers = nn.ModuleList()
        current_features = self.flatten_size
        
        for fc_size in fc_features:
            self.fc_layers.append(nn.Linear(current_features, fc_size))
            current_features = fc_size
        
        # Final classification layer
        self.fc_out = nn.Linear(current_features, num_classes)
        
        # Pooling and dropout layers
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_stride)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape [batch_size, channels, height, width]
            
        Returns:
        --------
        torch.Tensor
            Log softmax probabilities for each class
        """
        # Pass through sheafy convolutional layers
        for sheafy in self.sheafy_layers:
            x = sheafy(x)
            x = self.activation(x)
            x = self.pool(x)
        
        # Flatten for fully connected layers
        x = x.view(-1, self.flatten_size)
        
        # Pass through fully connected layers
        for fc in self.fc_layers:
            x = fc(x)
            x = self.activation(x)
            x = self.dropout(x)
        
        # Final classification
        x = self.fc_out(x)
        return F.log_softmax(x, dim=1)
    
    def print_model_info(self, name):
        """Print model architecture and parameter count"""
        print(f"\n{name}:")
        print(self)
        print(f"Total parameters: {sum(p.numel() for p in self.parameters()):,}")

    def save(self, path):
        """Save the model weights and architecture configuration"""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, path)
        
    @classmethod
    def load(cls, path):
        """Load a model with its weights and configuration"""
        checkpoint = torch.load(path)
        model = cls(**checkpoint['config'])
        model.load_state_dict(checkpoint['state_dict'])
        return model
    
    def save_weights(self, path):
        """Save only the model weights"""
        torch.save(self.state_dict(), path)
    
    def load_weights(self, path):
        """Load only the model weights"""
        self.load_state_dict(torch.load(path))
