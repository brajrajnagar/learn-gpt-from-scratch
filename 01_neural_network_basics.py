"""
=============================================================================
LESSON 1: Neural Network Basics - Building a Text Predictor
=============================================================================

This lesson builds ONE complete example: A simple text predictor.
We'll predict the next word in a sentence, step by step.

EXAMPLE FLOW: "The cat ___" → predict "sat"

By the end, you'll understand how GPT uses these same concepts!
"""

import numpy as np

# =============================================================================
# THE EXAMPLE WE'LL BUILD: Predicting next word in "The cat ___"
# =============================================================================

print("\n" + "="*70)
print("BUILDING: Next Word Predictor")
print("="*70)
print("""
GOAL: Given "The cat", predict the next word.

Possible completions:
- "The cat sat"     ← likely
- "The cat slept"   ← likely  
- "The cat ate"     ← possible
- "The cat quantum" ← unlikely!

HOW WE'LL BUILD THIS:
1. Single neuron → learns one pattern
2. Layer of neurons → learns multiple patterns
3. Multiple layers → learns complex patterns
4. This is exactly how GPT works!
""")

# =============================================================================
# STEP 1: Single Neuron - Learning ONE Pattern
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Single Neuron")
print("="*70)

def single_neuron(inputs, weights, bias):
    """
    A neuron learns to detect ONE pattern.
    
    OUR EXAMPLE: Detect if the sentence is about an ANIMAL doing something.
    
    INPUTS (features we extracted):
    - Is "cat" present? (1=yes, 0=no)
    - Is "dog" present?
    - Is "the" present?
    
    WEIGHTS (what the neuron learned):
    - "cat" → strong positive (animals are relevant!)
    - "dog" → strong positive (also animals!)
    - "the" → neutral (just grammar)
    
    OUTPUT: High = animal-related, Low = not animal-related
    """
    # Step 1: Weighted sum = how much does input match learned pattern?
    weighted_sum = np.dot(inputs, weights) + bias
    
    # Step 2: ReLU activation = only fire if pattern detected
    # (like a light switch: off if negative, on if positive)
    output = max(0, weighted_sum)
    
    return output

print("\n--- Let's Train a Neuron ---")
print("-"*50)
print("TASK: Detect if sentence is about an ANIMAL")
print()

# The neuron's LEARNED weights (after training)
# It learned: cat=important, dog=important, the=neutral
learned_weights = np.array([0.8, 0.7, 0.1])  # [cat, dog, the]
learned_bias = 0.1

print("Neuron learned these weights:")
print(f"  'cat' weight: {learned_weights[0]} ← important for animal detection")
print(f"  'dog' weight: {learned_weights[1]} ← important for animal detection")
print(f"  'the' weight: {learned_weights[2]} ← just grammar, not important")
print(f"  bias: {learned_bias} ← baseline activation")

print("\n" + "-"*50)
print("TEST 1: Input = 'cat the'")
print("-"*50)

# Input: "The cat" → [cat=1, dog=0, the=1]
inputs_cat = np.array([1.0, 0.0, 1.0])
output_cat = single_neuron(inputs_cat, learned_weights, learned_bias)

print(f"Input:  {inputs_cat}  ← [cat present, dog absent, the present]")
print(f"Weights: {learned_weights}")
print(f"Dot product: {np.dot(inputs_cat, learned_weights):.2f}")
print(f"After +bias: {np.dot(inputs_cat, learned_weights) + learned_bias:.2f}")
print(f"After ReLU:  {output_cat:.2f} ← Neuron FIRES! Detected animal context.")

print("\n" + "-"*50)
print("TEST 2: Input = 'car the'")
print("-"*50)

# Input: "The car" → [cat=0, dog=0, the=1]
inputs_car = np.array([0.0, 0.0, 1.0])
output_car = single_neuron(inputs_car, learned_weights, learned_bias)

print(f"Input:  {inputs_car}  ← [cat absent, dog absent, the present]")
print(f"After ReLU:  {output_car:.2f} ← Neuron SILENT. No animal detected.")

print("\n" + "-"*50)
print("KEY INSIGHT:")
print("-"*50)
print("""
A single neuron learns ONE pattern:
→ This neuron learned: "Is this about an animal?"

The computation flow:
1. INPUT:  [cat=1, dog=0, the=1]
2. MATCH:  Compare input to learned weights
3. SCORE:  Dot product = how well it matches
4. BIAS:   Adjust baseline
5. ACTIVATE: ReLU fires if pattern detected

GPT uses millions of neurons, each detecting different patterns!
""")

# =============================================================================
# STEP 2: Layer of Neurons - Learning MULTIPLE Patterns
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Dense Layer (Multiple Neurons)")
print("="*70)

def dense_layer(inputs, weights_matrix, biases):
    """
    A layer = multiple neurons working together.
    
    OUR EXAMPLE: Predict next word type after "The cat"
    
    We have 4 neurons, each predicting a DIFFERENT word category:
    - Neuron 1: Predicts ACTION words (sat, slept, ate, ran)
    - Neuron 2: Predicts LOCATION words (mat, couch, floor, bed)
    - Neuron 3: Predicts DESCRIPTOR words (lazy, hungry, sleepy, black)
    - Neuron 4: Predicts OTHER nouns (food, toy, mouse, ball)
    
    Each neuron has its OWN weights for the same inputs!
    """
    # All neurons compute in parallel (matrix multiplication)
    weighted_sum = np.dot(inputs, weights_matrix) + biases
    
    # ReLU activation for all neurons
    output = np.maximum(0, weighted_sum)
    
    return output

print("\n--- Building a 4-Neuron Layer ---")
print("-"*50)
print("TASK: Predict what TYPE of word comes next after 'The cat'")
print()

# Input: Features from "The cat"
# [cat_present, animal_context, article_present]
inputs = np.array([1.0, 1.0, 1.0])

print(f"Input from 'The cat': {inputs}")
print("  → cat is present")
print("  → context is animal-related")
print("  → 'the' (article) is present")

# Weights matrix: Each COLUMN is one neuron's weights
# Shape: (3 inputs, 4 neurons)
weights = np.array([
    # Action  Location  Descriptor  Other
    [ 0.6,     0.2,      0.3,        0.4],    # cat weight → each neuron
    [ 0.5,     0.1,      0.2,        0.3],    # animal context → each neuron
    [ 0.1,     0.3,      0.1,        0.2],    # article weight → each neuron
])

print("\nWeights matrix (each column = one neuron):")
print("            Action  Location  Descriptor  Other")
print(f"cat        {weights[0,0]:.1f}     {weights[0,1]:.1f}       {weights[0,2]:.1f}         {weights[0,3]:.1f}")
print(f"animal     {weights[1,0]:.1f}     {weights[1,1]:.1f}       {weights[1,2]:.1f}         {weights[1,3]:.1f}")
print(f"article    {weights[2,0]:.1f}     {weights[2,1]:.1f}       {weights[2,2]:.1f}         {weights[2,3]:.1f}")

biases = np.array([0.2, 0.1, 0.15, 0.1])
print(f"\nBiases: {biases}")

# Run the layer
output = dense_layer(inputs, weights, biases)

print("\n" + "="*50)
print("LAYER OUTPUT:")
print("="*50)
print(f"Input:  'The cat'")
print(f"Output: {output}")
print()
print("Interpretation:")
print(f"  Neuron 1 (Action):     {output[0]:.2f} ← {'HIGH - expects verb!' if output[0] > 0.5 else 'low'}")
print(f"  Neuron 2 (Location):   {output[1]:.2f} ← {'HIGH - expects place!' if output[1] > 0.5 else 'low'}")
print(f"  Neuron 3 (Descriptor): {output[2]:.2f} ← {'HIGH - expects adjective!' if output[2] > 0.5 else 'low'}")
print(f"  Neuron 4 (Other):      {output[3]:.2f} ← {'HIGH - expects noun!' if output[3] > 0.5 else 'low'}")

print("\n" + "-"*50)
print("WHAT HAPPENED:")
print("-"*50)
print("""
The layer processed 'The cat' and produced predictions:

Neuron 1: 'After "The cat", I expect an ACTION'
        → Output: 1.4 (high confidence!)
        
Neuron 2: 'Maybe a LOCATION?'
        → Output: 0.6 (possible)
        
Neuron 3: 'Maybe a DESCRIPTOR?'
        → Output: 0.65 (possible)
        
Neuron 4: 'Or another NOUN?'
        → Output: 0.9 (possible)

GPT does this at massive scale:
→ 50,000 output neurons (one per vocabulary word)
→ Each predicts probability for its word!
""")

# =============================================================================
# STEP 3: Multiple Layers - Learning Complex Patterns
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Deep Network (Multiple Layers)")
print("="*70)

class TextPredictorNetwork:
    """
    Multiple layers = learning complex patterns step by step.
    
    OUR EXAMPLE: Complete text prediction pipeline
    
    INPUT: "The cat" (encoded as features)
    
    LAYER 1: Detect basic patterns
    → "cat + the = animal subject"
    → "cat alone = pet context"
    
    LAYER 2: Combine into concepts
    → "animal subject + present tense = action expected"
    → "pet context + article = location expected"
    
    LAYER 3: Make final prediction
    → "sat" (most likely)
    → "slept" (second likely)
    → "ate" (possible)
    """
    
    def __init__(self):
        """
        Build a 3-layer prediction network.
        
        Architecture: 3 → 4 → 3 → 5
        
        INPUT (3): [cat, animal_context, article]
        
        HIDDEN 1 (4): Basic pattern detectors
        → "animal subject", "action expected", etc.
        
        HIDDEN 2 (3): Complex concept combiners
        → "subject-verb relationship", etc.
        
        OUTPUT (5): Word predictions
        → Probability for: sat, slept, ate, ran, meowed
        """
        np.random.seed(42)
        
        # Layer configurations
        self.layer1_w = np.random.randn(3, 4) * 0.5
        self.layer1_b = np.zeros(4)
        
        self.layer2_w = np.random.randn(4, 3) * 0.5
        self.layer2_b = np.zeros(3)
        
        self.layer3_w = np.random.randn(3, 5) * 0.5
        self.layer3_b = np.zeros(5)
        
        print("\n" + "="*50)
        print("NETWORK ARCHITECTURE:")
        print("="*50)
        print("Input (3) → Hidden1 (4) → Hidden2 (3) → Output (5)")
        print()
        print("  📥 INPUT: [cat, animal_context, article]")
        print("  ↓")
        print("  🏭 HIDDEN1: 4 pattern detectors")
        print("  ↓")
        print("  🏭 HIDDEN2: 3 concept combiners")
        print("  ↓")
        print("  📤 OUTPUT: 5 word probabilities")
        print("     [sat, slept, ate, ran, meowed]")
        print("="*50)
    
    def forward(self, inputs):
        """
        Pass input through all layers - the prediction flow.
        
        This is EXACTLY how GPT works, just smaller scale!
        """
        print("\n" + "="*50)
        print("FORWARD PASS: Predicting next word")
        print("="*50)
        
        x = inputs
        print(f"\n📥 INPUT: {x}")
        print("   Features from 'The cat'")
        
        # Layer 1
        x = np.dot(x, self.layer1_w) + self.layer1_b
        x = np.maximum(0, x)  # ReLU
        print(f"\n🏭 LAYER 1 OUTPUT: {np.round(x, 3)}")
        print("   Basic patterns detected")
        
        # Layer 2
        x = np.dot(x, self.layer2_w) + self.layer2_b
        x = np.maximum(0, x)  # ReLU
        print(f"\n🏭 LAYER 2 OUTPUT: {np.round(x, 3)}")
        print("   Complex concepts formed")
        
        # Layer 3 (output)
        x = np.dot(x, self.layer3_w) + self.layer3_b
        # No ReLU on output - we want raw scores (logits)
        print(f"\n📤 OUTPUT LOGITS: {np.round(x, 3)}")
        print("   Raw scores for each word")
        
        # Convert to probabilities (softmax)
        exp_x = np.exp(x - np.max(x))
        probs = exp_x / np.sum(exp_x)
        print(f"\n📊 PROBABILITIES: {np.round(probs, 4)}")
        
        return probs

print("\n--- Building Complete Predictor ---")
print("-"*50)

# Create network
predictor = TextPredictorNetwork()

# Input: "The cat"
inputs = np.array([1.0, 1.0, 1.0])

# Run prediction
print("\n" + "="*50)
print(f"PREDICTING: 'The cat ___'")
print("="*50)

probs = predictor.forward(inputs)

# Show results
words = ["sat", "slept", "ate", "ran", "meowed"]
print("\n" + "="*50)
print("PREDICTION RESULTS:")
print("="*50)
for word, prob in zip(words, probs):
    bar = "█" * int(prob * 20)
    print(f"  '{word}': {prob*100:5.1f}% {bar}")

best_word = words[np.argmax(probs)]
print(f"\n🎯 PREDICTED NEXT WORD: '{best_word}'")
print(f"   Sentence: 'The cat {best_word}...'")

# =============================================================================
# SUMMARY: How This Relates to GPT
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Connection to GPT")
print("="*70)

print("""
WHAT WE BUILT:
==============
1. Single neuron → Detected ONE pattern (animal context)
2. Layer of neurons → Detected MULTIPLE patterns (action, location, etc.)
3. Multiple layers → Combined patterns into predictions
4. Output → Word probabilities

HOW GPT IS THE SAME:
====================
✓ GPT uses neurons (billions of them!)
✓ GPT uses layers (many transformer blocks)
✓ GPT uses forward pass (input → layers → output)
✓ GPT outputs word probabilities

HOW GPT IS DIFFERENT:
=====================
→ Specialized layers (attention, not just dense)
→ Much bigger (millions of neurons per layer)
→ Trained on massive data (entire internet)
→ Learns context (not just word features)

THE CORE IDEA (same for both):
==============================
INPUT → LAYER1 → LAYER2 → ... → OUTPUT
         ↓        ↓
      weights  weights
      learned  learned

Next: 02_embeddings.py - How do we convert "The cat" to numbers?
=============================================================================""")

print("\n" + "="*70)
print("EXERCISE: Try Different Inputs")
print("="*70)

print("""
Try these inputs and see how predictions change:

1. "The dog" → [0, 1, 1] (no cat, but still animal)
   predictor.forward(np.array([0.0, 1.0, 1.0]))

2. "The car" → [0, 0, 1] (no animal context)
   predictor.forward(np.array([0.0, 0.0, 1.0]))

3. "The cat dog" → [1, 1, 1] (both animals)
   predictor.forward(np.array([1.0, 1.0, 1.0]))

Notice how the network's predictions change based on input!
This is the foundation of all language models.
=============================================================================""")