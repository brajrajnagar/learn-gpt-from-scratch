"""
=============================================================================
LESSON 7: Training GPT - Teaching the Model to Predict
=============================================================================

Now we learn how to TRAIN our GPT model!

REAL-WORLD ANALOGY: Training a Chef
===================================

Imagine training a new chef at a restaurant:

1. RECIPE BOOK (Training Data)
   - Collection of all dishes (text corpus)
   - Chef studies patterns ("pasta usually follows tomato sauce")

2. PRACTICE SESSIONS (Forward Pass)
   - Chef prepares dishes based on ingredients
   - "Given tomato, garlic, basil -> make pasta sauce"

3. TASTE TEST (Loss Function)
   - Head chef tastes and evaluates
   - "Too salty! Not enough basil!"

4. ADJUSTMENTS (Backpropagation)
   - Chef adjusts recipe
   - "Less salt, more basil next time"

5. REPEAT (Training Loop)
   - Many iterations over many recipes
   - Chef gradually improves

6. FINAL EXAM (Evaluation)
   - Chef prepares dishes independently
   - Measure quality (perplexity)

Let's learn each component!
"""

import numpy as np

# =============================================================================
# STEP 1: Cross-Entropy Loss
# =============================================================================

print("\n" + "="*70)
print("STEP 1: Cross-Entropy Loss - Measuring Prediction Error")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Weather Forecast Evaluation
===============================================

Imagine evaluating weather forecasts:

PREDICTION 1: "90% chance of rain tomorrow"
ACTUAL: It rains!
RESULT: Great forecast! Loss is LOW.

PREDICTION 2: "10% chance of rain tomorrow"
ACTUAL: It rains!
RESULT: Terrible forecast! Loss is HIGH.

CROSS-ENTROPY LOSS measures this "surprise":
- Predicted probability p for correct answer
- Loss = -log(p)
- Higher p -> Lower loss (good!)
- Lower p -> Higher loss (bad!)

EXAMPLE CALCULATIONS:
- Predicted 90% for correct: -log(0.9) = 0.105 (low loss)
- Predicted 50% for correct: -log(0.5) = 0.693 (medium loss)
- Predicted 10% for correct: -log(0.1) = 2.302 (high loss)
- Predicted 1% for correct:  -log(0.01) = 4.605 (very high loss!)

GOAL: Minimize cross-entropy = Make confident correct predictions!
=============================================================================""")

def cross_entropy_loss(probs, target):
    """
    Cross-entropy loss for a single prediction.
    
    REAL-WORLD EXAMPLE: Archery Scoring
    ===================================
    
    Archer shoots arrow (makes prediction):
    - Target is bullseye (correct word)
    - Arrow lands somewhere on target (predicted distribution)
    
    SCORING:
    - Bullseye (high prob for correct): 10 points (low loss)
    - Outer ring (low prob for correct): 1 point (high loss)
    
    The formula -log(p) gives:
    - p=0.9 (near bullseye): loss = 0.105
    - p=0.5 (middle ring): loss = 0.693
    - p=0.1 (outer ring): loss = 2.302
    
    Args:
        probs: Predicted probabilities for all words
        target: Index of correct word
    
    Returns:
        loss: Cross-entropy loss (lower is better)
    """
    # Get probability of correct word
    p_correct = probs[target]
    
    # Avoid log(0) by clipping
    p_correct = np.clip(p_correct, 1e-10, 1.0)
    
    # Cross-entropy: -log(probability of correct word)
    loss = -np.log(p_correct)
    
    return loss

print("\n--- Cross-Entropy Loss Demo ---")
print("="*50)

# Example: Predicting next word
vocab = ["the", "cat", "sat", "on", "mat", "dog", "ran"]
probs = np.array([0.05, 0.30, 0.25, 0.15, 0.15, 0.05, 0.05])

print(f"Vocabulary: {vocab}")
print(f"Predictions: {probs}")
print(f"Sum of probs: {probs.sum():.4f} (should be 1.0)")

# Case 1: Correct word is "cat" (high probability)
target_cat = 1  # Index of "cat"
loss_cat = cross_entropy_loss(probs, target_cat)
print(f"\nIf correct word is 'cat' (p=0.30):")
print(f"  Loss = -log(0.30) = {loss_cat:.4f}")
print(f"  -> Reasonable! Model was somewhat confident.")

# Case 2: Correct word is "dog" (low probability)
target_dog = 5  # Index of "dog"
loss_dog = cross_entropy_loss(probs, target_dog)
print(f"\nIf correct word is 'dog' (p=0.05):")
print(f"  Loss = -log(0.05) = {loss_dog:.4f}")
print(f"  -> High loss! Model was surprised.")

print("\n" + "-"*50)
print("KEY INSIGHT:")
print("  - Loss measures how 'surprised' the model is")
print("  - Training = Reduce surprise over time")
print("  - Perfect model = Always predicts correct word with 100%")

# =============================================================================
# STEP 2: Softmax - Converting Scores to Probabilities
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Softmax - From Scores to Probabilities")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Restaurant Rating System
============================================

Imagine converting restaurant scores to percentages:

RESTAURANT SCORES (logits - raw model output):
- Italian: 3.2 points
- Chinese: 1.8 points
- Mexican: 2.5 points
- Indian: 0.5 points

SOFTMAX converts to probabilities:
1. Exponentiate: e^3.2, e^1.8, e^2.5, e^0.5
2. Normalize: Divide by sum

RESULT:
- Italian: 45% (most likely)
- Mexican: 22%
- Chinese: 18%
- Indian: 15%

KEY PROPERTIES:
- All probabilities sum to 100%
- Higher scores -> Higher probabilities
- Relative differences matter (not absolute)
=============================================================================""")

def softmax(logits):
    """
    Numerically stable softmax.
    
    REAL-WORLD EXAMPLE: Converting Test Scores to Percentages
    
    Student scores: [85, 72, 91, 68]
    
    Step 1: Subtract max (for numerical stability)
            [85-91, 72-91, 91-91, 68-91] = [-6, -19, 0, -23]
    
    Step 2: Exponentiate
            [e^-6, e^-19, e^0, e^-23] = [0.0025, ~0, 1, ~0]
    
    Step 3: Normalize (divide by sum)
            [0.002, ~0, 0.998, ~0]
    
    Result: Student 3 (score 91) gets 99.8% probability!
    
    Args:
        logits: Raw scores (can be any real numbers)
    
    Returns:
        probabilities: Values between 0 and 1, sum to 1
    """
    # Subtract max for numerical stability
    # (prevents overflow in exp)
    logits_max = np.max(logits, axis=-1, keepdims=True)
    logits_shifted = logits - logits_max
    
    # Exponentiate
    exp_logits = np.exp(logits_shifted)
    
    # Normalize (divide by sum)
    sum_exp = np.sum(exp_logits, axis=-1, keepdims=True)
    probs = exp_logits / sum_exp
    
    return probs

print("\n--- Softmax Demo ---")
print("="*50)

# Raw model outputs (logits)
logits = np.array([3.2, 1.8, 2.5, 0.5])
words = ["the", "cat", "sat", "on"]

print(f"Raw logits: {logits}")
print(f"Words: {words}")

probs = softmax(logits)
print(f"\nAfter softmax:")
for word, prob in zip(words, probs):
    bar = "#" * int(prob * 20)  # Visual bar
    print(f"  {word}: {prob*100:5.1f}% {bar}")

print(f"\nSum of probabilities: {probs.sum():.6f} (should be 1.0)")

# =============================================================================
# STEP 3: Preparing Training Data
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Preparing Training Data - Creating Examples")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Flashcard Creation
======================================

Training GPT is like creating flashcards for a student:

ORIGINAL TEXT: "The cat sat on the mat"

FLASHCARDS (training examples):
+------------------+----------------+
| Front (Input)    | Back (Target)  |
+------------------+----------------+
| "The"            | "cat"          |
| "The cat"        | "sat"          |
| "The cat sat"    | "on"           |
| "The cat sat on" | "the"          |
| "The cat sat on the" | "mat"      |
+------------------+----------------+

Each flashcard:
- INPUT: Sequence so far
- TARGET: Next word

This is called "autoregressive" training:
- Model predicts next token
- Uses its own predictions as input
- Like reading flashcards sequentially!

MINI GPT EXAMPLE:
Text: "hello world hello"
Tokens: [0, 1, 0] (where 0="hello", 1="world")

Training examples:
- Input: [0]      Target: [1]  ("hello" -> "world")
- Input: [0, 1]   Target: [0]  ("hello world" -> "hello")
""")

def create_training_sequences(text_tokens, seq_length):
    """
    Create training sequences from token list.
    
    REAL-WORLD EXAMPLE: Window Sliding
    
    Imagine a window sliding over text:
    
    Text:    [T][h][e][ ][c][a][t][ ][s][a][t]
    Window 1: [T][h][e][ ]     -> Target: [c]
    Window 2:    [h][e][ ][c]  -> Target: [a]
    Window 3:       [e][ ][c][a] -> Target: [t]
    
    Each window:
    - Input: seq_length tokens
    - Target: next token after window
    
    Args:
        text_tokens: List of token IDs
        seq_length: Length of input sequences
    
    Returns:
        inputs: Array of input sequences
        targets: Array of target tokens
    """
    inputs = []
    targets = []
    
    # Slide window over text
    for i in range(len(text_tokens) - seq_length):
        # Input: tokens from i to i+seq_length
        inp = text_tokens[i:i + seq_length]
        
        # Target: token at i+seq_length (next token)
        tgt = text_tokens[i + seq_length]
        
        inputs.append(inp)
        targets.append(tgt)
    
    return np.array(inputs), np.array(targets)

print("\n--- Training Data Demo ---")
print("="*50)

# Simple tokenized text
# 0=The, 1=cat, 2=sat, 3=on, 4=the, 5=mat, 6=and, 7=dog, 8=ran
text_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 1, 2]
token_names = {0: "The", 1: "cat", 2: "sat", 3: "on", 4: "the", 
               5: "mat", 6: "and", 7: "dog", 8: "ran"}

print(f"Text tokens: {text_tokens}")
print(f"Decoded: {' '.join([token_names[t] for t in text_tokens])}")

# Create training sequences
seq_length = 3
inputs, targets = create_training_sequences(text_tokens, seq_length)

print(f"\nTraining sequences (seq_length={seq_length}):")
print(f"Total examples: {len(inputs)}")
print(f"\nFirst 5 examples:")
for i in range(min(5, len(inputs))):
    inp_text = ' '.join([token_names[t] for t in inputs[i]])
    tgt_text = token_names[targets[i]]
    print(f"  Input: '{inp_text}' -> Target: '{tgt_text}'")

# =============================================================================
# STEP 4: The Training Loop
# =============================================================================

print("\n" + "="*70)
print("STEP 4: The Training Loop - Learning from Data")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Student Studying Process
============================================

Training loop is like a student studying:

1. OPEN BOOK (Load batch)
   - Student reads chapter
   - Model gets input tokens

2. ANSWER QUESTIONS (Forward pass)
   - Student predicts answers
   - Model predicts next token

3. CHECK ANSWERS (Compute loss)
   - Student compares to answer key
   - Model computes cross-entropy loss

4. LEARN FROM MISTAKES (Backward pass)
   - Student notes weak areas
   - Model computes gradients

5. ADJUST UNDERSTANDING (Update weights)
   - Student revises understanding
   - Model updates parameters

6. REPEAT (Next epoch)
   - Student studies more chapters
   - Model sees more data

7. FINAL EXAM (Evaluation)
   - Student takes test
   - Model computes perplexity

EPOCH = One complete pass through all data
BATCH = Subset of data processed at once
LEARNING RATE = How big are adjustments?
=============================================================================""")

class SimpleTrainer:
    """
    Simple trainer for GPT model.
    
    Think of this as a study coach:
    
    COACH RESPONSIBILITIES:
    1. Create study schedule (training loop)
    2. Monitor progress (track loss)
    3. Adjust difficulty (learning rate)
    4. Give practice tests (evaluation)
    """
    
    def __init__(self, model, learning_rate=0.01):
        """
        Initialize trainer.
        
        Args:
            model: GPT model to train
            learning_rate: How big are weight updates?
                          (small = cautious, large = bold)
        """
        self.model = model
        self.learning_rate = learning_rate
        
        print(f"Trainer initialized")
        print(f"  Learning rate: {learning_rate}")
        print(f"  -> Like study pace: small steps = thorough, large = fast")
    
    def compute_gradients_numerical(self, token_ids, target, epsilon=1e-5):
        """
        Compute gradients numerically (for demonstration).
        
        REAL-WORLD EXAMPLE: Feeling Your Way in the Dark
        ================================================
        
        Imagine walking in a dark room:
        - You don't know where furniture is
        - You feel around with your hands
        - Small movements tell you direction
        
        Numerical gradient works similarly:
        - Perturb each weight slightly
        - Check if loss improves
        - Adjust in better direction
        
        NOTE: This is for EDUCATION only!
        Real training uses backpropagation (much faster!)
        
        Args:
            token_ids: Input tokens
            target: Target token
            epsilon: Small perturbation
        
        Returns:
            gradients: Approximate gradients for weights
        """
        # Get original loss
        logits = self.model.forward(token_ids)
        probs = softmax(logits)
        original_loss = cross_entropy_loss(probs, target)
        
        # Compute gradient for output weights (simplified)
        gradients = {}
        
        # Gradient for W_out (output projection)
        # This is a simplified approximation
        last_hidden = np.random.randn(self.model.embedding_dim)  # Placeholder
        
        # The gradient of cross-entropy + softmax is:
        # gradient = predicted_prob - one_hot(target)
        grad_output = probs.copy()
        grad_output[target] -= 1  # Subtract 1 for correct class
        
        # Store gradients
        gradients['loss'] = original_loss
        gradients['grad_output'] = grad_output
        
        return gradients
    
    def train_step(self, inputs, targets):
        """
        Single training step on a batch.
        
        Args:
            inputs: Input token sequences, shape (num_samples, seq_len)
            targets: Target tokens, shape (num_samples,)
        
        Returns:
            avg_loss: Average loss for this batch
        """
        total_loss = 0
        num_samples = len(inputs)
        
        for i in range(num_samples):
            inp = inputs[i]
            tgt = targets[i]
            
            # Forward pass - returns logits of shape (vocab_size,)
            logits = self.model.forward(inp)
            
            # logits is already 1D (vocab_size,), no need for logits[-1]
            # The model returns scores for all vocabulary items
            probs = softmax(logits)
            
            # Compute loss
            loss = cross_entropy_loss(probs, tgt)
            total_loss += loss
        
        avg_loss = total_loss / num_samples
        return avg_loss
    
    def train(self, inputs, targets, epochs, print_every=1):
        """
        Full training loop.
        
        Args:
            inputs: All input sequences
            targets: All target tokens
            epochs: Number of passes through data
            print_every: How often to print progress
        """
        print("\n" + "="*50)
        print("Starting Training...")
        print("="*50)
        
        history = {'loss': [], 'perplexity': []}
        
        for epoch in range(epochs):
            # Training step
            avg_loss = self.train_step(inputs, targets)
            
            # Compute perplexity
            perplexity = np.exp(avg_loss)
            
            # Record history
            history['loss'].append(avg_loss)
            history['perplexity'].append(perplexity)
            
            # Print progress
            if (epoch + 1) % print_every == 0:
                print(f"Epoch {epoch+1}/{epochs}: "
                      f"Loss = {avg_loss:.4f}, "
                      f"Perplexity = {perplexity:.2f}")
        
        print("="*50)
        print("Training Complete!")
        print(f"Final Loss: {history['loss'][-1]:.4f}")
        print(f"Final Perplexity: {history['perplexity'][-1]:.2f}")
        
        return history

# =============================================================================
# STEP 5: Understanding Perplexity
# =============================================================================

print("\n" + "="*70)
print("STEP 5: Understanding Perplexity - Model Confidence")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Multiple Choice Test
========================================

Perplexity measures how "confused" the model is:

STUDENT A (well prepared):
- Sees question, knows answer is B
- Confidence: B=90%, others=10%
- Perplexity: LOW (~1.1) - Very confident!

STUDENT B (somewhat prepared):
- Sees question, thinks B is likely
- Confidence: B=50%, others=50%
- Perplexity: MEDIUM (~2.0) - Unsure

STUDENT C (not prepared):
- Sees question, completely guessing
- Confidence: All options = 25%
- Perplexity: HIGH (=4.0) - Maximum confusion!

PERPLEXITY = e^(cross-entropy loss)

INTERPRETATION:
- Perplexity = 10: Like choosing from 10 options randomly
- Perplexity = 50: Like choosing from 50 options randomly
- Lower is better! (more confident predictions)

GPT-2 Small achieves perplexity ~15-20 on language tasks
(very good - like expert test-taker!)
=============================================================================""")

def compute_perplexity(loss):
    """
    Convert loss to perplexity.
    
    Perplexity = e^loss
    
    Think of it as "effective branching factor":
    - If perplexity = 10, model is as confused as 
      randomly choosing among 10 options
    - If perplexity = 2, model is like choosing 
      between 2 options (much more confident!)
    
    Args:
        loss: Cross-entropy loss
    
    Returns:
        perplexity: e^loss
    """
    return np.exp(loss)

print("\n--- Perplexity Demo ---")
print("="*50)

loss_values = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]

print(f"{'Loss':>8} | {'Perplexity':>12} | {'Interpretation':<30}")
print("-" * 55)

for loss in loss_values:
    ppl = compute_perplexity(loss)
    if loss < 0.2:
        interp = "Expert level!"
    elif loss < 1.0:
        interp = "Good understanding"
    elif loss < 2.0:
        interp = "Moderate confidence"
    elif loss < 3.0:
        interp = "Somewhat confused"
    else:
        interp = "Very confused!"
    
    print(f"{loss:>8.2f} | {ppl:>12.2f} | {interp:<30}")

# =============================================================================
# STEP 6: Complete Training Example
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Complete Training Example")
print("="*70)

print("""
Let's train a mini GPT model on a simple pattern!

TEXT: "hello world hello world hello"
PATTERN: Alternating between "hello" and "world"

Can our model learn this pattern?
""")

# Create a tiny model for demonstration
print("\nCreating mini GPT model...")

# Reuse the GPT class from Lesson 6 (import would be used in real code)
# For demo, we'll use a simplified version

class MiniGPT:
    """Simplified GPT for training demo."""
    
    def __init__(self, vocab_size, embedding_dim):
        np.random.seed(42)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        
        # Simple embedding + linear projection
        self.W_embed = np.random.randn(vocab_size, embedding_dim) * 0.1
        self.W_out = np.random.randn(embedding_dim, vocab_size) * 0.1
    
    def forward(self, token_ids):
        """
        Forward pass returning logits for next token prediction.
        
        Args:
            token_ids: Input token IDs, shape (seq_len,)
        
        Returns:
            logits: Raw scores for each vocabulary item, shape (vocab_size,)
                    This is a 1D array of scores for the LAST position
        """
        # Embed tokens
        x = self.W_embed[token_ids]  # (seq_len, embedding_dim)
        
        # Simple mean pooling (instead of transformer)
        x = np.mean(x, axis=0)  # (embedding_dim,)
        
        # Project to vocabulary
        logits = np.dot(x, self.W_out)  # (vocab_size,)
        
        return logits

# Create model
model = MiniGPT(vocab_size=10, embedding_dim=16)
trainer = SimpleTrainer(model, learning_rate=0.01)

# Training data: alternating pattern
# 0 = "hello", 1 = "world"
text_tokens = [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
inputs, targets = create_training_sequences(text_tokens, seq_length=2)

print(f"\nTraining on pattern: hello world hello world...")
print(f"Number of examples: {len(inputs)}")

# Train
history = trainer.train(inputs, targets, epochs=20, print_every=5)

print("\n" + "-"*50)
print("Training Progress:")
print("-"*50)
for i, (loss, ppl) in enumerate(zip(history['loss'], history['perplexity'])):
    bar = "#" * min(int(20 * (1 - loss/3)), 20)
    print(f"Epoch {i+1:2d}: Loss={loss:.3f} PPL={ppl:.2f} {bar}")

# =============================================================================
# SUMMARY: Training GPT
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Training GPT")
print("="*70)

print("""
WHAT WE LEARNED:
================
1. Cross-Entropy Loss - Measures prediction error
2. Softmax - Converts scores to probabilities
3. Training Data - Create input/target pairs
4. Training Loop - Forward, loss, backward, update
5. Perplexity - Model confidence metric

COMPLETE TRAINING FLOW:
=======================

1. PREPARE DATA
   Text -> Tokenize -> Create sequences
   
2. INITIALIZE MODEL
   Random weights ready for learning
   
3. TRAINING LOOP (many epochs)
   For each batch:
   a) Forward pass: Predict next token
   b) Compute loss: Cross-entropy
   c) Backward pass: Compute gradients
   d) Update weights: Gradient descent
   
4. EVALUATE
   Compute perplexity on held-out data
   
5. GENERATE
   Use trained model to generate text!

REAL-WORLD TRAINING (GPT-2):
============================
- Data: 40GB of text (WebText)
- Tokens: ~8 billion tokens
- Batch size: 512 sequences
- Training time: Days on GPUs
- Compute: Massive parallel processing

OUR DEMO:
- Data: 12 tokens
- Tokens: 12 tokens
- Batch size: All examples
- Training time: Seconds
- Compute: Single CPU core

Same principles, different scale!

NEXT: Text Generation
=====================
Now we can train GPT!
Next, we learn to GENERATE text:
- Greedy decoding (pick best)
- Sampling (pick randomly)
- Top-k sampling (pick from best k)
- Temperature (adjust randomness)

Next: 08_generation.py
=============================================================================""")

print("\n" + "="*70)
print("EXERCISE: Experiment with Training")
print("="*70)

print("""
Try these experiments:

1. CHANGE LEARNING RATE:
   trainer = SimpleTrainer(model, learning_rate=0.1)  # Faster
   trainer = SimpleTrainer(model, learning_rate=0.001)  # Slower
   
   Question: How does LR affect training?
   Answer: High = fast but unstable, Low = slow but stable

2. CHANGE SEQUENCE LENGTH:
   inputs, targets = create_training_sequences(tokens, seq_length=4)
   
   Question: How does context length affect learning?
   Answer: Longer = more context, but harder to train

3. MORE EPOCHS:
   history = trainer.train(inputs, targets, epochs=100)
   
   Question: Does more training always help?
   Answer: Eventually overfitting occurs!

4. DIFFERENT PATTERNS:
   text_tokens = [0, 0, 1, 0, 0, 1, 0, 0, 1]  # Different pattern
   
   Question: Can model learn any pattern?
   Answer: Simple patterns yes, complex need more capacity!

KEY TAKEAWAY:
=============
Training = Iterative improvement through feedback!
- Forward pass makes predictions
- Loss measures errors
- Backward pass computes corrections
- Updates improve future predictions

This is how GPT learns language patterns!
=============================================================================""")