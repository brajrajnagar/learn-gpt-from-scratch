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
    
    REAL-WORLD EXAMPLE: Spam Email Detector
    ----------------------------------------
    Imagine you're building a spam filter. This neuron detects if an email is spam.
    
    Inputs: Features of the email
    - How many exclamation marks (!!!) 
    - Does it contain "$" (money-related)?
    - Does it contain "FREE"?
    
    Weights: How important each feature is for detecting spam
    - Many exclamations → likely spam (positive weight)
    - Dollar signs → likely spam (positive weight)
    - "FREE" → very likely spam (high positive weight)
    
    Bias: Base tendency to classify as spam (even with zero features)
    
    Output: 
    - High value = likely spam
    - Low/zero value = not spam
    
    Args:
        inputs: Input values (e.g., [0.8, 0.6, -0.3])
        weights: Weight for each input (e.g., [0.3, 0.7, -0.5])
        bias: Single bias value (e.g., 0.1)
    
    Returns:
        The output of the neuron after applying ReLU activation
    """
    # Step 1: Compute weighted sum (dot product)
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Step 2: Apply activation function (ReLU)
    # ReLU: Returns 0 if negative, otherwise returns the value
    # Why ReLU? It introduces non-linearity - the network can learn complex patterns
    output = max(0, weighted_sum)
    
    return output

# REAL-WORLD EXAMPLE: Spam Email Detector
print("\n--- Single Neuron: Spam Email Detector ---")
print("="*50)
print("""
SCENARIO: You're building a spam email classifier.

This neuron has learned to detect spam emails based on 3 features:
1. Number of exclamation marks in subject (!!!, more = spammy)
2. Presence of dollar signs ($$$, money-related = spammy)
3. Word "FREE" (very spammy indicator)

The neuron's weights represent what it has LEARNED:
- Exclamation marks: weight = 0.5 (somewhat spammy)
- Dollar signs: weight = 0.7 (quite spammy)
- "FREE": weight = -0.4 (actually, this example shows negative)

Let's see it in action!
""")

print("\nExample 1: SPAMMY email")
print("-"*50)
# Email with lots of exclamations and dollar signs
spammy_email_features = np.array([0.8, 0.6, -0.3])  
# [exclamations!, dollar$, contains_FREE]
spam_weights = np.array([0.5, 0.7, -0.4])  
spam_bias = 0.1

spam_score = single_neuron(spammy_email_features, spam_weights, spam_bias)
print(f"Email features: {spammy_email_features}")
print(f"  - Exclamation marks: 0.8 (many!!!)")
print(f"  - Dollar signs: 0.6 (some $$$)")
print(f"  - Contains 'FREE': -0.3 (no)")
print(f"\nNeuron weights (learned patterns): {spam_weights}")
print(f"Bias: {spam_bias}")
print(f"\nWeighted sum: {np.dot(spammy_email_features, spam_weights) + spam_bias:.4f}")
print(f"Spam score (after ReLU): {spam_score:.4f}")
print(f"→ High score = likely SPAM!")

print("\nExample 2: NORMAL email")
print("-"*50)
# Normal email with no spammy features
normal_email_features = np.array([0.1, 0.0, 0.0])  
normal_score = single_neuron(normal_email_features, spam_weights, spam_bias)
print(f"Email features: {normal_email_features}")
print(f"  - Exclamation marks: 0.1 (few)")
print(f"  - Dollar signs: 0.0 (none)")
print(f"  - Contains 'FREE': 0.0 (no)")
print(f"\nSpam score (after ReLU): {normal_score:.4f}")
print(f"→ Low score = likely NOT spam!")

print("\nExample 3: VERY spammy email")
print("-"*50)
# Extremely spammy email
very_spammy = np.array([1.0, 1.0, 1.0])
very_spam_score = single_neuron(very_spammy, spam_weights, spam_bias)
print(f"Email features: {very_spammy}")
print(f"  - All spam indicators maxed out!")
print(f"\nSpam score (after ReLU): {very_spam_score:.4f}")
print(f"→ Definitely SPAM! Delete it!")

# =============================================================================
# STEP 2: Neural Network Layer (Multiple Neurons)
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Neural Network Layer (Multiple Neurons)")
print("="*70)

def dense_layer(inputs, weights_matrix, biases):
    """
    A layer contains multiple neurons. Each neuron produces one output value.
    
    REAL-WORLD EXAMPLE: Multi-Feature Text Classifier
    --------------------------------------------------
    Imagine classifying movie reviews. You have 3 input features:
    - Word "amazing" frequency
    - Word "boring" frequency  
    - Word "excellent" frequency
    
    This layer has 4 neurons, each detecting different patterns:
    - Neuron 1: Detects positive sentiment
    - Neuron 2: Detects negative sentiment
    - Neuron 3: Detects mixed feelings
    - Neuron 4: Detects sarcasm
    
    Each neuron has its own set of weights (learned differently).
    
    Args:
        inputs: Input vector of shape (input_size,)
        weights_matrix: Shape (input_size, output_size) - each column is one neuron's weights
        biases: Bias for each neuron, shape (output_size,)
    
    Returns:
        Output of shape (output_size,) - one value per neuron
    """
    # Matrix multiplication: combines inputs with all neurons at once
    # This is efficient - all neurons compute in parallel!
    weighted_sum = np.dot(inputs, weights_matrix) + biases
    
    # Apply ReLU activation to all outputs
    # Neurons that detected patterns fire (positive), others stay silent (0)
    output = np.maximum(0, weighted_sum)
    
    return output

print("\n--- Dense Layer: Movie Review Classifier ---")
print("="*50)
print("""
SCENARIO: You're building a movie review sentiment analyzer.

INPUT: Word frequencies from a review
- Frequency of "amazing"
- Frequency of "boring"
- Frequency of "excellent"

LAYER: 4 neurons, each detecting different sentiment patterns:
- Neuron 1: Positive sentiment detector
- Neuron 2: Negative sentiment detector
- Neuron 3: Neutral/mixed detector
- Neuron 4: Enthusiasm level detector

OUTPUT: 4 values representing different sentiment aspects
""")

# Input: Word frequencies from review "This movie was amazing and excellent!"
inputs = np.array([0.5, -0.2, 0.8])
print(f"\nReview: 'This movie was amazing and excellent!'")
print(f"Input features: {inputs}")
print(f"  - 'amazing' frequency: 0.5")
print(f"  - 'boring' frequency: -0.2 (not present)")
print(f"  - 'excellent' frequency: 0.8")

# Weights: Each of 4 neurons has 3 weights (one per input word)
# Shape: (3 inputs, 4 neurons)
weights = np.array([
    [0.3, -0.1, 0.7, 0.2],   # "amazing" weights → each neuron
    [-0.2, 0.4, 0.2, -0.3],  # "boring" weights → each neuron
    [0.5, -0.3, -0.1, 0.6],  # "excellent" weights → each neuron
])
print(f"\nWeights matrix shape: {weights.shape}")
print(f"Each column = one neuron's weights for all 3 words")

biases = np.array([0.1, -0.2, 0.3, 0.05])
print(f"Biases: {biases}")
print(f"  - Neuron 1 bias: 0.1 (slightly positive-leaning)")
print(f"  - Neuron 2 bias: -0.2 (slightly negative-leaning)")
print(f"  - Neuron 3 bias: 0.3 (neutral baseline)")
print(f"  - Neuron 4 bias: 0.05 (enthusiasm baseline)")

output = dense_layer(inputs, weights, biases)

print(f"\n{'='*50}")
print("LAYER OUTPUT:")
print(f"{'='*50}")
print(f"Output shape: {output.shape}")
print(f"Output values: {output}")
print(f"\nInterpretation:")
print(f"  - Neuron 1 (positive): {output[0]:.4f} ← DETECTED!")
print(f"  - Neuron 2 (negative): {output[1]:.4f} ← Not activated")
print(f"  - Neuron 3 (neutral):  {output[2]:.4f} ← DETECTED!")
print(f"  - Neuron 4 (enthusiasm): {output[3]:.4f} ← DETECTED!")
print(f"\nConclusion: This is a POSITIVE review with enthusiasm!")

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
    
    REAL-WORLD EXAMPLE: Complete Sentiment Analysis Pipeline
    ---------------------------------------------------------
    Think of this as a factory assembly line:
    
    Layer 1 (Input → Hidden1): Raw features → Basic patterns
    - Takes raw word frequencies
    - Detects simple combinations like "amazing + excellent = very positive"
    
    Layer 2 (Hidden1 → Hidden2): Basic patterns → Complex concepts
    - Combines basic patterns into higher-level concepts
    - e.g., "positive words + no negatives = strong recommendation"
    
    Layer 3 (Hidden2 → Output): Complex concepts → Final decision
    - Maps complex concepts to final classification
    - e.g., "strong recommendation + enthusiasm = 5-star rating"
    
    Each layer transforms the data into a more abstract representation!
    """
    
    def __init__(self, layer_sizes):
        """
        Initialize the network.
        
        REAL-WORLD EXAMPLE: Factory Setup
        ----------------------------------
        layer_sizes = [3, 8, 4, 2] means:
        
        STATION 1 (Input): 3 raw features
        - Word frequencies: "amazing", "boring", "excellent"
        
        STATION 2 (Hidden1): 8 workers detecting basic patterns
        - Each worker looks for different word combinations
        
        STATION 3 (Hidden2): 4 supervisors detecting complex concepts
        - Each supervisor combines patterns from workers
        
        STATION 4 (Output): 2 final classifications
        - "Positive review" or "Negative review"
        
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
        
        print(f"\n{'='*50}")
        print("FACTORY SETUP:")
        print(f"{'='*50}")
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization: helps training
            # This scaling prevents signals from exploding or vanishing
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * scale
            b = np.zeros(layer_sizes[i+1])
            self.weights.append(w)
            self.biases.append(b)
            
            print(f"Station {i+1}: {layer_sizes[i]} inputs → {layer_sizes[i+1]} workers")
        
        print(f"\nTotal layers (stations): {len(layer_sizes) - 1}")
        print(f"{'='*50}")
        
    def forward(self, inputs):
        """
        Forward propagation: Pass input through all layers.
        
        REAL-WORLD EXAMPLE: Assembly Line
        ----------------------------------
        Data flows through the network like a product on an assembly line:
        
        1. Raw materials (input) enter Station 1
        2. Station 1 processes and passes to Station 2
        3. Station 2 further processes and passes to Station 3
        4. Station 3 produces final product (output)
        
        At each station:
        - Workers (neurons) examine the input
        - They apply their expertise (weights)
        - They make decisions (activation)
        - They pass results to next station
        
        Args:
            inputs: Input array (raw materials)
        
        Returns:
            Final output after passing through all layers (finished product)
        """
        current_output = inputs
        
        print(f"\n{'='*50}")
        print("ASSEMBLY LINE IN ACTION:")
        print(f"{'='*50}")
        print(f"\n📦 Raw materials (input): {current_output}")
        print(f"   Shape: {current_output.shape}")
        
        # Pass through each layer (station)
        for i, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            print(f"\n--- Station {i+1} Processing ---")
            
            # Linear transformation: workers apply their expertise
            current_output = np.dot(current_output, weights) + biases
            print(f"   After linear transform: {np.round(current_output, 4)}")
            
            # Apply ReLU activation (except possibly the last layer)
            # ReLU is like a filter - only positive signals pass through
            if i < len(self.weights) - 1:  # Not the last layer
                current_output = np.maximum(0, current_output)
                print(f"   After ReLU filter: {np.round(current_output, 4)}")
            
            print(f"   Output shape: {current_output.shape}")
            print(f"   → Passed to {'next station' if i < len(self.weights)-2 else 'final output'}")
        
        return current_output

print("\n--- Deep Network: Complete Sentiment Pipeline ---")
print("="*70)
print("""
SCENARIO: Building a complete sentiment analysis pipeline.

NETWORK ARCHITECTURE: [3, 8, 4, 2]

📥 INPUT (3 features):
   Word frequencies from review: ["amazing", "boring", "excellent"]

🏭 HIDDEN LAYER 1 (8 workers):
   Detect basic patterns:
   - "amazing + excellent" together
   - "boring" alone
   - Any single positive word
   - Combinations...

🏭 HIDDEN LAYER 2 (4 supervisors):
   Detect complex concepts:
   - "Overwhelmingly positive"
   - "Mixed feelings"
   - "Clearly negative"
   - "Enthusiastic recommendation"

📤 OUTPUT (2 classifications):
   - Probability of "Positive review"
   - Probability of "Negative review"
""")

# Create network: Input(3) → Hidden1(8) → Hidden2(4) → Output(2)
network = SimpleNeuralNetwork([3, 8, 4, 2])

# Run forward propagation with a sample review
inputs = np.array([0.5, -0.2, 0.8])
print(f"\n{'='*70}")
print("RUNNING SENTIMENT ANALYSIS:")
print(f"{'='*70}")
print(f"\nReview: 'The movie was amazing and excellent!'")
print(f"Input features: {inputs}")
print(f"  - 'amazing': 0.5")
print(f"  - 'boring': -0.2 (not present)")
print(f"  - 'excellent': 0.8")

output = network.forward(inputs)

print(f"\n{'='*70}")
print("FINAL CLASSIFICATION:")
print(f"{'='*70}")
print(f"\nOutput: {output}")
print(f"  - Positive score: {output[0]:.4f}")
print(f"  - Negative score: {output[1]:.4f}")

if output[0] > output[1]:
    print(f"\n✅ PREDICTION: POSITIVE REVIEW!")
else:
    print(f"\n❌ PREDICTION: NEGATIVE REVIEW!")

# =============================================================================
# STEP 4: Understanding Why Layers Matter for GPT
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Connection to GPT")
print("="*70)

print("""
REAL-WORLD EXAMPLE: How GPT Uses These Concepts
================================================

GPT is essentially a VERY deep neural network with specialized layers.

📥 INPUT LAYER:
   - Takes token embeddings (768-dimensional vectors for each word)
   - Example: "The cat sat" → [embedding_for_"The", embedding_for_"cat", ...]
   
🏭 TRANSFORMER BLOCKS (many layers):
   - GPT-2 Small: 12 transformer blocks (12 "super-stations")
   - GPT-3: Up to 96 transformer blocks
   
   Each transformer block contains:
   - Self-Attention: "Which words should I pay attention to?"
     Example: In "The cat sat on the mat because it was tired"
     → "it" should attend to "cat" (not "mat")
   
   - Feed-Forward: Process the information
     Similar to our dense_layer but more sophisticated
   
   - Layer Norm: Stabilize the signals (like quality control)
   
📤 OUTPUT LAYER:
   - Produces probabilities for next token prediction
   - Size = vocabulary size (e.g., 50,000 possible tokens)
   - Example: After "The cat sat on the", predict:
     * "mat" (30% chance)
     * "couch" (15% chance)
     * "floor" (10% chance)
     * ...

THE KEY INSIGHT:
================
Despite all the complexity, GPT fundamentally uses:
  OUTPUT = ACTIVATION(DOT_PRODUCT(INPUT, WEIGHTS) + BIAS)

Just like our simple neuron! The difference is:
- BILLIONS of parameters instead of dozens
- Specialized architectures (attention) for sequence understanding
- Massive training data to learn patterns

=============================================================================""")

# =============================================================================
# EXERCISE: Try it yourself!
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Modify and Experiment")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS TO TRY:
==============================

1. SPAM DETECTOR EXPERIMENT:
   Change the email features and see spam scores change:
   
   spammy_email = [1.0, 1.0, 1.0]  # Maximum spam indicators
   normal_email = [0.0, 0.0, 0.0]  # No spam indicators
   suspicious_email = [0.5, 0.5, 0.0]  # Some red flags
   
   Try: spam_score = single_neuron(suspicious_email, spam_weights, spam_bias)

2. MOVIE REVIEW EXPERIMENT:
   Create reviews with different sentiments:
   
   positive_review = [0.8, 0.0, 0.9]  # "amazing" and "excellent"
   negative_review = [0.0, 0.9, 0.0]  # "boring"
   mixed_review = [0.5, 0.5, 0.5]  # All words present
   
   Try: output = dense_layer(positive_review, weights, biases)

3. DEEP NETWORK EXPERIMENT:
   Create networks with different architectures:
   
   # Deeper network (more layers = more abstract thinking)
   deep_net = SimpleNeuralNetwork([3, 16, 8, 4, 2])
   
   # Wider network (more neurons per layer = more pattern detection)
   wide_net = SimpleNeuralNetwork([3, 32, 32, 2])
   
   # Tiny network (simpler but faster)
   tiny_net = SimpleNeuralNetwork([3, 4, 2])

4. COUNT PARAMETERS:
   How many learnable parameters does a network have?
   
   For each layer: parameters = (input_size × output_size) + output_size
   (weights + biases)
   
   For [3, 8, 4, 2]:
   - Layer 1: (3×8) + 8 = 32 parameters
   - Layer 2: (8×4) + 4 = 36 parameters
   - Layer 3: (4×2) + 2 = 10 parameters
   - TOTAL: 78 parameters
   
   GPT-3 has 175 BILLION parameters!

KEY TAKEAWAY:
=============
✓ Neural networks transform input data through layers
✓ Each layer applies: weights × input + bias, then activation
✓ Multiple layers = hierarchical feature learning
✓ GPT uses these same principles with specialized attention layers
✓ The "deep" in deep learning = many layers of abstraction

Next: In 02_embeddings.py, we'll learn how words become vectors!
=============================================================================""")