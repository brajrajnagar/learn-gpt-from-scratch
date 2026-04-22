"""
=============================================================================
LESSON 1: Neural Network Basics - The Foundation of GPT
=============================================================================

Before building GPT, we need to understand the basic building block: 
the neural network layer. This file teaches you:

1. What is a neuron?
2. What is a layer?
3. Forward propagation - how data flows through the network
4. Activation functions - adding non-linearity

KEY CONCEPTS:
- Neuron: A computational unit that takes inputs, applies weights, adds bias, 
  and produces an output
- Layer: A collection of neurons working together
- Weights: Learnable parameters that determine the strength of connections
- Bias: Learnable parameter that shifts the activation function
- Activation Function: Introduces non-linearity (ReLU, Sigmoid, etc.)

Let's build a simple neural network from scratch!
"""

import numpy as np

# =============================================================================
# STEP 1: Understanding a Single Neuron
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Single Neuron")
print("="*70)

def single_neuron(inputs, weights, bias):
    """
    A single neuron does this:
    output = activation(dot_product(inputs, weights) + bias)
    
    Args:
        inputs: Input values (e.g., [0.5, -0.2, 0.8])
        weights: Weight for each input (e.g., [0.3, 0.7, -0.5])
        bias: Single bias value (e.g., 0.1)
    
    Returns:
        The output of the neuron after applying ReLU activation
    """
    # Step 1: Compute weighted sum (dot product)
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Step 2: Apply activation function (ReLU)
    # ReLU: Returns 0 if negative, otherwise returns the value
    output = max(0, weighted_sum)
    
    return output

# Example: A neuron that might detect "positive sentiment" in text
print("\n--- Single Neuron Example ---")
print("Imagine this neuron detects positive sentiment from word scores")

# Input: scores for 3 words (e.g., "good", "great", "bad")
inputs = np.array([0.8, 0.6, -0.3])  # "good" and "great" are positive, "bad" is negative
weights = np.array([0.5, 0.7, -0.4])  # Neuron learned that first two indicate positivity
bias = 0.1

output = single_neuron(inputs, weights, bias)
print(f"Inputs (word scores): {inputs}")
print(f"Weights: {weights}")
print(f"Bias: {bias}")
print(f"Weighted Sum: {np.dot(inputs, weights) + bias:.4f}")
print(f"Output after ReLU: {output:.4f}")

# =============================================================================
# STEP 2: Neural Network Layer (Multiple Neurons)
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Neural Network Layer")
print("="*70)

def dense_layer(inputs, weights_matrix, biases):
    """
    A layer contains multiple neurons. Each neuron produces one output value.
    
    Args:
        inputs: Input vector of shape (input_size,)
        weights_matrix: Shape (input_size, output_size) - each column is one neuron's weights
        biases: Bias for each neuron, shape (output_size,)
    
    Returns:
        Output of shape (output_size,) - one value per neuron
    """
    # Matrix multiplication: combines inputs with all neurons at once
    weighted_sum = np.dot(inputs, weights_matrix) + biases
    
    # Apply ReLU activation to all outputs
    output = np.maximum(0, weighted_sum)
    
    return output

print("\n--- Dense Layer Example ---")
print("A layer with 4 neurons, each receiving 3 inputs")

# Input: 3 values (e.g., 3 word embeddings)
inputs = np.array([0.5, -0.2, 0.8])

# Weights: Each of 4 neurons has 3 weights (one per input)
# Shape: (3 inputs, 4 neurons)
# Each row represents weights for one input feature across all neurons
weights = np.array([
    [0.3, -0.1, 0.7, 0.2],   # Input 1 weights to neurons 1,2,3,4
    [-0.2, 0.4, 0.2, -0.3],  # Input 2 weights to neurons 1,2,3,4
    [0.5, -0.3, -0.1, 0.6],  # Input 3 weights to neurons 1,2,3,4
])  # Shape: (3, 4) - 3 inputs, 4 neurons

# Biases: One per neuron
biases = np.array([0.1, -0.2, 0.3, 0.05])

output = dense_layer(inputs, weights, biases)
print(f"Input shape: {inputs.shape}")
print(f"Weights shape: {weights.shape}")
print(f"Biases shape: {biases.shape}")
print(f"Output shape: {output.shape}")
print(f"Output values: {output}")

# =============================================================================
# STEP 3: Multiple Layers (Deep Neural Network)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Deep Neural Network (Multiple Layers)")
print("="*70)

class SimpleNeuralNetwork:
    """
    A simple neural network with multiple layers.
    This is the foundation of more complex models like GPT.
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize the network.
        
        Args:
            layer_sizes: List of sizes for each layer
                        e.g., [3, 8, 4, 2] means:
                        - Input layer: 3 neurons
                        - Hidden layer 1: 8 neurons
                        - Hidden layer 2: 4 neurons
                        - Output layer: 2 neurons
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Create random weights and biases for each layer
        np.random.seed(42)  # For reproducible results
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization: helps training
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
        
        print(f"Created network with layers: {layer_sizes}")
        print(f"Number of layers: {len(layer_sizes) - 1}")
        
    def forward(self, inputs):
        """
        Forward propagation: Pass input through all layers.
        
        Args:
            inputs: Input array
        
        Returns:
            Final output after passing through all layers
        """
        current_output = inputs
        
        print(f"\n  Input shape: {current_output.shape}")
        
        # Pass through each layer
        for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            # Linear transformation
            current_output = np.dot(current_output, weights) + biases
            
            # Apply ReLU activation (except possibly the last layer)
            if i < len(self.weights) - 1:  # Not the last layer
                current_output = np.maximum(0, current_output)
            
            print(f"  After Layer {i+1}: shape = {current_output.shape}")
        
        return current_output

print("\n--- Deep Network Example ---")
print("Building a network: 3 → 8 → 4 → 2")

# Create network: Input(3) → Hidden1(8) → Hidden2(4) → Output(2)
network = SimpleNeuralNetwork([3, 8, 4, 2])

# Run forward propagation
inputs = np.array([0.5, -0.2, 0.8])
print(f"\nForward pass with input: {inputs}")
output = network.forward(inputs)
print(f"\nFinal output: {output}")

# =============================================================================
# STEP 4: Understanding Why Layers Matter for GPT
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Connection to GPT")
print("="*70)

print("""
In GPT, we use the same basic principles:

1. INPUT LAYER:
   - Takes token embeddings (e.g., 768-dimensional vectors representing words)
   
2. HIDDEN LAYERS:
   - GPT uses many transformer blocks (each contains multiple layers)
   - GPT-2 Small: 12 transformer blocks
   - GPT-3: Up to 96 transformer blocks
   
3. OUTPUT LAYER:
   - Produces probabilities for next token prediction
   - Size = vocabulary size (e.g., 50,000 tokens)

The key difference in GPT:
- Instead of simple dense layers, GPT uses:
  * Self-Attention layers (learn relationships between words)
  * Feed-Forward layers (process the information)
  * Layer Normalization (stabilizes training)
  
But the fundamental concept is the same:
  OUTPUT = ACTIVATION(DOT_PRODUCT(INPUT, WEIGHTS) + BIAS)

=============================================================================""")

# =============================================================================
# EXERCISE: Try it yourself!
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Modify and Experiment")
print("="*70)

print("""
Try these experiments:

1. Change the input values and see how output changes:
   inputs = np.array([1.0, 0.5, -0.5])

2. Create a deeper network:
   network = SimpleNeuralNetwork([3, 16, 8, 4, 2])

3. Create a wider network:
   network = SimpleNeuralNetwork([3, 32, 32, 2])

4. Count total parameters:
   Total = sum of (input_size * output_size + output_size) for each layer

Key Takeaway:
- Neural networks transform input data through layers
- Each layer applies: weights * input + bias, then activation
- GPT is built from these same principles, just more sophisticated!

Next: In 02_embeddings.py, we'll learn how words become vectors!
=============================================================================""")