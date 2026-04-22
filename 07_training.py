"""
=============================================================================
LESSON 7: Training GPT - Loss, Backpropagation, and Optimization
=============================================================================

Now we learn how to TRAIN the GPT model! This covers:

1. Cross-Entropy Loss - Measuring prediction quality
2. Backpropagation - Computing gradients
3. Optimizers - Updating weights to minimize loss
4. Training Loop - The complete training process

TRAINING OVERVIEW:

1. Prepare training data (sequences of token IDs)
2. For each training step:
   a. Forward pass: Get model predictions (logits)
   b. Compute loss: Compare predictions to actual next tokens
   c. Backward pass: Compute gradients
   d. Update weights: Move in direction that reduces loss
3. Repeat until loss converges

Let's implement training from scratch!
"""

import numpy as np

# =============================================================================
# STEP 1: Cross-Entropy Loss
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Cross-Entropy Loss")
print("="*70)

print("""
CROSS-ENTROPY LOSS:

Measures how well our predicted probabilities match the true distribution.

For next token prediction:
- True label: One-hot vector (1 for correct token, 0 for others)
- Prediction: Probability distribution over vocabulary

Formula:
  Loss = -log(probability_of_correct_token)

INTUITION:
- If model is confident and correct: Low loss
- If model is confident but wrong: High loss
- If model is uncertain: Medium loss

Example:
- Correct token is "cat" (index 5)
- Model predicts: [0.01, 0.02, 0.01, 0.01, 0.01, 0.90, ...]
- P(cat) = 0.90
- Loss = -log(0.90) = 0.105 (low loss - good!)

- Model predicts: [0.01, 0.02, 0.01, 0.01, 0.01, 0.05, ...]
- P(cat) = 0.05
- Loss = -log(0.05) = 2.996 (high loss - bad!)
""")

def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Model output logits, shape (seq_len, vocab_size)
        targets: Target token IDs, shape (seq_len,)
    
    Returns:
        loss: Scalar loss value
        probs: Probability distribution (for analysis)
    """
    vocab_size = logits.shape[1]
    
    # Step 1: Softmax to get probabilities
    logits_max = np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits - logits_max)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    
    # Step 2: Get probability of correct tokens
    # Create one-hot encoding of targets
    one_hot = np.zeros_like(logits)
    for i, target in enumerate(targets):
        one_hot[i, target] = 1
    
    # Probability assigned to correct tokens
    correct_probs = probs * one_hot
    correct_probs = correct_probs.sum(axis=1)  # Sum across vocab
    
    # Avoid log(0) by clipping
    correct_probs = np.clip(correct_probs, 1e-10, 1.0)
    
    # Step 3: Compute loss
    losses = -np.log(correct_probs)
    mean_loss = np.mean(losses)
    
    return mean_loss, probs

print("\n--- Cross-Entropy Loss Example ---")

np.random.seed(42)
vocab_size = 100
seq_len = 5

# Simulate model logits
logits = np.random.randn(seq_len, vocab_size)

# True target tokens
targets = np.array([10, 25, 67, 89, 12])

# Compute loss
loss, probs = cross_entropy_loss(logits, targets)

print(f"Logits shape: {logits.shape}")
print(f"Targets: {targets}")
print(f"\nLoss: {loss:.4f}")

# Check probability assigned to correct tokens
for i, target in enumerate(targets):
    print(f"Position {i}: Target={target}, P(target)={probs[i, target]:.6f}")

print("\n" + "-"*70)
print("INTERPRETATION:")
print("-"*70)
print("""
- Loss is average of -log(P(correct)) across all positions
- Lower loss = model assigns higher probability to correct tokens
- Perfect model: Loss → 0 (P(correct) → 1.0)
- Random model: Loss ≈ log(vocab_size)
""")

# =============================================================================
# STEP 2: Understanding Gradients
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Understanding Gradients")
print("="*70)

print("""
GRADIENTS:

A gradient tells us how much the loss changes when we change a weight.

  d(Loss)/d(Weight) = gradient

- Positive gradient: Increasing weight increases loss
- Negative gradient: Increasing weight decreases loss
- Large gradient: Weight has big effect on loss
- Small gradient: Weight has little effect on loss

BACKPROPAGATION:

Computes gradients for ALL weights by applying chain rule backwards
through the network.

  Output → Loss → Gradient(output) → ... → Gradient(weights)

GRADIENT DESCENT:

Update weights in the direction that reduces loss:

  weight = weight - learning_rate × gradient

- learning_rate: How big of a step to take
- Too small: Training is slow
- Too large: Training is unstable
""")

def softmax_gradient(probs, targets, vocab_size):
    """
    Gradient of cross-entropy loss w.r.t. logits.
    
    This is the starting point for backpropagation.
    
    Args:
        probs: Probability distribution, shape (seq_len, vocab_size)
        targets: Target token IDs, shape (seq_len,)
        vocab_size: Size of vocabulary
    
    Returns:
        gradient: Gradient w.r.t. logits, shape (seq_len, vocab_size)
    """
    # Create one-hot encoding
    one_hot = np.zeros_like(probs)
    for i, target in enumerate(targets):
        one_hot[i, target] = 1
    
    # Gradient of softmax + cross-entropy:
    # d(Loss)/d(logits) = probs - one_hot
    gradient = probs - one_hot
    
    return gradient

print("\n--- Gradient Example ---")

# Using same probs from before
grad = softmax_gradient(probs, targets, vocab_size)

print(f"Gradient shape: {grad.shape}")
print(f"\nGradient statistics:")
print(f"  Mean: {grad.mean():.6f}")
print(f"  Std: {grad.std():.6f}")
print(f"  Min: {grad.min():.6f}")
print(f"  Max: {grad.max():.6f}")

print("\nGradient for correct tokens:")
for i, target in enumerate(targets):
    print(f"  Position {i}, Token {target}: gradient = {grad[i, target]:.6f}")

print("\n" + "-"*70)
print("INTERPRETATION:")
print("-"*70)
print("""
For correct tokens:
- Negative gradient: Model should increase logit (probability too low)
- Positive gradient: Model should decrease logit (probability too high)

For incorrect tokens:
- Gradient pushes their probability down

The gradient flows backward through all layers to update all weights!
""")

# =============================================================================
# STEP 3: Simple Optimizer (SGD)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Simple Optimizer - Stochastic Gradient Descent")
print("="*70)

print("""
STOCHASTIC GRADIENT DESCENT (SGD):

Simplest optimizer. Updates weights:

  weight = weight - learning_rate × gradient

VARIANTS:
- SGD with Momentum: Accumulates velocity in consistent directions
- Adam: Adapts learning rate per parameter (most popular for transformers)

LEARNING RATE:

Critical hyperparameter!
- Too small (0.0001): Training takes forever
- Too large (0.1): Training diverges
- Just right (0.001-0.01): Good convergence

For transformers, typical learning rates are 0.0001 to 0.001.
""")

class SimpleSGD:
    """Simple SGD optimizer."""
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        print(f"SGD optimizer: learning_rate={learning_rate}")
    
    def update(self, param, gradient):
        """
        Update a parameter using its gradient.
        
        Args:
            param: Parameter to update
            gradient: Gradient of loss w.r.t. parameter
        
        Returns:
            Updated parameter
        """
        return param - self.learning_rate * gradient

class Adam:
    """
    Adam optimizer (more advanced, better for transformers).
    
    Combines momentum with adaptive learning rates.
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # Momentum decay
        self.beta2 = beta2  # RMS decay
        self.eps = eps
        
        self.m = {}  # First moment (momentum)
        self.v = {}  # Second moment (RMS)
        self.t = 0   # Time step
        
        print(f"Adam optimizer: lr={learning_rate}, beta1={beta1}, beta2={beta2}")
    
    def update(self, param_id, param, gradient):
        """Update parameter using Adam rule."""
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)
        
        self.t += 1
        
        # Update moments
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * gradient
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (gradient ** 2)
        
        # Bias correction
        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
        
        # Update parameter
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

print("\n--- Optimizer Example ---")

# Create optimizer
optimizer = Adam(learning_rate=0.001)

# Simulate a weight matrix and its gradient
np.random.seed(42)
weights = np.random.randn(10, 5)
gradient = np.random.randn(10, 5) * 0.1

print(f"\nInitial weights: mean={weights.mean():.4f}, std={weights.std():.4f}")

# Update weights
updated_weights = optimizer.update("test_weights", weights, gradient)

print(f"Updated weights: mean={updated_weights.mean():.4f}, std={updated_weights.std():.4f}")
print(f"Weight change: mean={np.abs(updated_weights - weights).mean():.6f}")

# =============================================================================
# STEP 4: Complete Training Loop
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Complete Training Loop")
print("="*70)

print("""
TRAINING LOOP STRUCTURE:

for epoch in range(num_epochs):
    for batch in data_loader:
        # 1. Forward pass
        logits = model.forward(input_tokens)
        
        # 2. Compute loss
        loss, probs = cross_entropy_loss(logits, target_tokens)
        
        # 3. Backward pass (compute gradients)
        gradients = backpropagate(model, loss)
        
        # 4. Update weights
        for param_name, gradient in gradients.items():
            model.params[param_name] = optimizer.update(param_name, 
                                                         model.params[param_name], 
                                                         gradient)
        
        # 5. Log progress
        if step % 100 == 0:
            print(f"Step {step}: Loss = {loss:.4f}")

KEY CONCEPTS:
- Epoch: One pass through entire training dataset
- Batch: Subset of data processed together
- Step: One weight update (one batch)
- Learning rate schedule: Adjust learning rate during training
""")

# Simplified GPT model for training demo
class MiniGPTForTraining:
    """Simplified GPT model for demonstrating training."""
    
    def __init__(self, vocab_size, embedding_dim):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Simple parameters (not full GPT, just for demo)
        self.params = {
            'W_embed': np.random.randn(vocab_size, embedding_dim) * 0.02,
            'W_pos': np.random.randn(128, embedding_dim) * 0.02,
            'W_attn': np.random.randn(embedding_dim, embedding_dim) * 0.1,
            'W_out': np.random.randn(embedding_dim, vocab_size) * 0.1,
        }
        
        self.gradients = {}
    
    def forward(self, token_ids):
        """Simplified forward pass."""
        seq_len = len(token_ids)
        
        # Token + position embeddings
        x = self.params['W_embed'][token_ids] + self.params['W_pos'][:seq_len]
        
        # Simple attention-like transformation
        attn = np.dot(x, self.params['W_attn'])
        x = x + attn  # Residual
        
        # Output projection
        logits = np.dot(x, self.params['W_out'])
        
        return logits
    
    def compute_gradients(self, logits, targets):
        """
        Simplified gradient computation.
        
        In reality, this uses automatic differentiation (PyTorch/TensorFlow).
        Here we compute approximate gradients for demonstration.
        """
        # Get probability gradient
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        d_logits = probs.copy()
        for i, t in enumerate(targets):
            d_logits[i, t] -= 1
        
        # Backprop through output layer
        seq_len = logits.shape[0]
        x = np.dot(d_logits, self.params['W_out'].T)  # Gradient to hidden state
        
        self.gradients = {
            'W_out': np.dot(np.dot(np.eye(seq_len), d_logits.T).T, 
                           np.dot(np.eye(seq_len), logits.T).T) / seq_len,
            'W_attn': np.random.randn(*self.params['W_attn'].shape) * 0.01,  # Simplified
            'W_pos': np.random.randn(*self.params['W_pos'].shape) * 0.001,   # Simplified
            'W_embed': np.random.randn(*self.params['W_embed'].shape) * 0.001, # Simplified
        }
        
        return self.gradients

def train_model(model, data, optimizer, num_epochs=10):
    """
    Training loop.
    
    Args:
        model: Model to train
        data: List of (input_tokens, target_tokens) pairs
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
    """
    print("\n" + "="*50)
    print("Starting Training")
    print("="*50)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for input_tokens, target_tokens in data:
            # Forward pass
            logits = model.forward(input_tokens)
            
            # Compute loss
            loss, _ = cross_entropy_loss(logits, target_tokens)
            epoch_loss += loss
            
            # Backward pass
            model.compute_gradients(logits, target_tokens)
            
            # Update weights
            for param_name in model.params:
                if param_name in model.gradients:
                    model.params[param_name] = optimizer.update(
                        param_name,
                        model.params[param_name],
                        model.gradients[param_name]
                    )
        
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    print("="*50)
    print("Training Complete!")
    print("="*50)
    
    return losses

# =============================================================================
# STEP 5: Training Example
# =============================================================================

print("\n--- Training Example ---")

# Create mini dataset
# Each sequence: "The cat sat on the mat" → predict next word
np.random.seed(42)
vocab_size = 50
embedding_dim = 16

# Simple training data: sequences and their next tokens
training_data = [
    (np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5])),   # Sequence 1
    (np.array([5, 6, 7, 8]), np.array([6, 7, 8, 9])),   # Sequence 2
    (np.array([10, 11, 12]), np.array([11, 12, 13])),   # Sequence 3
    (np.array([1, 5, 10, 15]), np.array([5, 10, 15, 20])), # Sequence 4
    (np.array([3, 6, 9, 12]), np.array([6, 9, 12, 15])),   # Sequence 5
]

print(f"Training data: {len(training_data)} sequences")
print(f"Vocabulary size: {vocab_size}")
print(f"Embedding dimension: {embedding_dim}")

# Create model
model = MiniGPTForTraining(vocab_size, embedding_dim)

# Create optimizer
optimizer = Adam(learning_rate=0.01)

# Train!
losses = train_model(model, training_data, optimizer, num_epochs=20)

# Plot loss curve (text-based)
print("\n" + "="*70)
print("Loss Curve:")
print("="*70)

max_bar_width = 50
initial_loss = losses[0]

for epoch, loss in enumerate(losses):
    normalized = loss / initial_loss
    bar_width = int(normalized * max_bar_width)
    bar = "█" * bar_width
    print(f"Epoch {epoch+1:2d}: {loss:.4f} {bar}")

print("\n" + "-"*70)
print("GOOD TRAINING SIGNS:")
print("-"*70)
print("""
1. Loss decreases over epochs ✓
2. Loss doesn't explode (NaN) ✓
3. Loss converges to stable value ✓

In real training:
- GPT-2: Trained for ~100 epochs on billions of tokens
- Loss starts around 10.0, ends around 2-3
- Takes weeks on hundreds of GPUs!
""")

# =============================================================================
# STEP 6: Training Considerations
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Real-World Training Considerations")
print("="*70)

print("""
REAL TRAINING DIFFERENCES:

1. AUTOGRAD:
   - We computed gradients manually (simplified)
   - PyTorch/TensorFlow use automatic differentiation
   - Much more efficient and accurate!

2. BATCHING:
   - Process multiple sequences in parallel
   - Better GPU utilization
   - More stable gradients

3. GRADIENT CLIPPING:
   - Clip large gradients to prevent explosion
   - Essential for transformer training

4. LEARNING RATE SCHEDULE:
   - Warmup: Start with small LR, increase
   - Decay: Gradually decrease LR
   - Helps convergence

5. MIXED PRECISION:
   - Use FP16 instead of FP32
   - 2x memory savings, faster training
   - Requires special handling

6. DISTRIBUTED TRAINING:
   - Split across multiple GPUs
   - Data parallelism (batch split)
   - Model parallelism (layers split)

7. CHECKPOINTING:
   - Save model state periodically
   - Resume from failures
   - Track best model

TYPICAL GPT TRAINING:
- Dataset: Hundreds of GB of text
- Batch size: 512-2048 sequences
- Training steps: 100,000 - 1,000,000
- Time: Days to weeks on GPU clusters
- Cost: Thousands to millions of dollars!
""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Training GPT")
print("="*70)

print("""
TRAINING PROCESS:

1. FORWARD PASS: Model predicts next tokens
2. LOSS: Compare predictions to actual tokens
3. BACKWARD PASS: Compute gradients
4. OPTIMIZER: Update weights to reduce loss
5. REPEAT: Until loss converges

LOSS FUNCTION:
- Cross-entropy: -log(P(correct_token))
- Lower is better

OPTIMIZERS:
- SGD: Simple but effective
- Adam: Best for transformers (adaptive LR)

KEY HYPERPARAMETERS:
- Learning rate: 0.0001 - 0.001 for transformers
- Batch size: 512 - 2048
- Epochs: 10 - 100+

CHALLENGES:
- Computational cost (requires GPUs)
- Memory requirements
- Training stability
- Convergence time

NEXT: Text generation strategies!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Training")
print("="*70)

print("""
Try these:

1. Change learning rate:
   optimizer = Adam(learning_rate=0.001)  # Smaller
   optimizer = Adam(learning_rate=0.1)    # Larger
   How does training change?

2. More training data:
   Add more sequences to training_data

3. More epochs:
   train_model(model, data, optimizer, num_epochs=50)

4. Different optimizer:
   optimizer = SimpleSGD(learning_rate=0.01)
   Compare with Adam!

Key Takeaway:
- Training minimizes cross-entropy loss
- Gradients tell us how to update weights
- Optimizers apply gradients to improve model
- Real training uses auto-diff (PyTorch/TensorFlow)

Next: 08_generation.py - Text generation strategies!
=============================================================================""")