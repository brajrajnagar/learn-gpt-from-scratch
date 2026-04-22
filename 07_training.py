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
REAL-WORLD EXAMPLE: Weather Forecast Scoring
============================================

Imagine you're a weather forecaster predicting tomorrow's weather:

WEATHER OPTIONS: Sunny, Cloudy, Rainy, Stormy, Snowy

YOUR FORECAST (probability distribution):
- Sunny: 60%
- Cloudy: 25%
- Rainy: 10%
- Stormy: 4%
- Snowy: 1%

NEXT DAY ACTUAL WEATHER: Sunny ☀️

HOW GOOD WAS YOUR FORECAST?

Cross-entropy loss measures forecast accuracy:
  Loss = -log(P(correct_outcome))
  Loss = -log(0.60) = 0.51

INTERPRETATION:
- Loss = 0.51 → Pretty good forecast!
- You were confident (60%) and correct

WHAT IF YOU WERE WRONG?

Scenario A: Confident but Wrong
- Your forecast: Sunny 90%, Rainy 2%
- Actual: Rainy
- Loss = -log(0.02) = 3.91 ← HIGH PENALTY!
- Being confidently wrong is BAD!

Scenario B: Uncertain and Wrong
- Your forecast: Sunny 40%, Rainy 35%, Cloudy 25%
- Actual: Stormy (you gave 0%!)
- Loss = -log(0.001) = 6.91 ← VERY HIGH!
- Never assign 0% probability!

KEY INSIGHTS:
==============

1. PERFECT CONFIDENCE = PERFECT SCORE
   If you're 100% sure and right: Loss = -log(1.0) = 0

2. BEING CONFIDENTLY WRONG IS COSTLY
   If you're 99% sure and wrong: Loss = -log(0.01) = 4.6

3. UNCERTAINTY IS SAFER
   If you spread probabilities: Loss is moderate either way

4. NEVER PREDICT ZERO
   -log(0) = infinity! (That's why we clip to 1e-10)

TRAINING ANALOGY:
=================

Think of training like a student studying for exams:

STUDENT = GPT Model
EXAM QUESTIONS = Training data
STUDENT'S ANSWERS = Model predictions
CORRECT ANSWERS = Target tokens
GRADE = Cross-entropy loss

GOOD STUDYING:
- Student predicts "Paris" for "Capital of France"
- Correct answer: Paris
- Loss = -log(0.95) = 0.05 ← Low loss!
- Student studied well!

BAD STUDYING:
- Student predicts "London" for "Capital of France"
- Correct answer: Paris
- Loss = -log(0.01) = 4.60 ← High loss!
- Student needs to study more!

TRAINING = STUDENT LEARNING FROM MISTAKES!
Each training step adjusts weights to reduce future loss.
=============================================================================""")

def cross_entropy_loss(logits, targets):
    """
    Compute cross-entropy loss.
    
    REAL-WORLD EXAMPLE: Restaurant Quality Score
    ============================================
    
    Imagine a restaurant rating system:
    
    CUSTOMER ORDERS: [Appetizer, Salad, Main, Dessert]
    CHEF'S PREDICTIONS: Probability for each dish
    
    After dinner, compare:
    - What customer actually ordered
    - What chef predicted they'd order
    
    Loss = Average surprise across all courses
    
    Low loss = Chef knows customer's preferences
    High loss = Chef needs to learn customer's taste!
    
    Args:
        logits: Model output logits, shape (seq_len, vocab_size)
        targets: Target token IDs, shape (seq_len,)
    
    Returns:
        loss: Scalar loss value (lower = better predictions)
        probs: Probability distribution (for analysis)
    """
    vocab_size = logits.shape[1]
    
    # Step 1: Softmax to get probabilities
    # Convert raw scores to probabilities (like weather forecast %)
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
    
    # Avoid log(0) by clipping (never predict 0%!)
    correct_probs = np.clip(correct_probs, 1e-10, 1.0)
    
    # Step 3: Compute loss
    # -log(P) for each position, then average
    losses = -np.log(correct_probs)
    mean_loss = np.mean(losses)
    
    return mean_loss, probs

print("\n--- Cross-Entropy Loss Example ---")
print("="*50)
print("""
SCENARIO: Weather forecasting for 5 days

Model predicts probabilities for each day
Actual weather is known (targets)

Let's see how good the predictions were!
""")

np.random.seed(42)
vocab_size = 100
seq_len = 5

# Simulate model logits (unnormalized scores)
logits = np.random.randn(seq_len, vocab_size)

# True target tokens (what actually happened)
targets = np.array([10, 25, 67, 89, 12])

# Compute loss
loss, probs = cross_entropy_loss(logits, targets)

print(f"Model predictions analyzed:")
print(f"  Logits shape: {logits.shape}")
print(f"  Targets (actual outcomes): {targets}")
print(f"\n📊 OVERALL LOSS: {loss:.4f}")
print(f"  → Average 'surprise' across all predictions")
print(f"  → Lower = model was more confident about correct answers")

print(f"\n📈 BREAKDOWN BY POSITION:")
for i, target in enumerate(targets):
    prob_assigned = probs[i, target]
    position_loss = -np.log(np.clip(prob_assigned, 1e-10, 1.0))
    confidence = "🎯" if prob_assigned > 0.5 else "🤔" if prob_assigned > 0.1 else "😕"
    print(f"  Day {i+1}: Actual={target}, Predicted P={prob_assigned:.4f}, Loss={position_loss:.2f} {confidence}")

print("\n" + "-"*70)
print("INTERPRETATION:")
print("-"*70)
print("""
🎯 GOOD PREDICTION (P > 0.5):
   Model was confident AND correct
   → Low contribution to loss

🤔 UNCERTAIN PREDICTION (0.1 < P < 0.5):
   Model wasn't sure
   → Moderate contribution to loss

😕 BAD PREDICTION (P < 0.1):
   Model was surprised (confident but wrong, or very uncertain)
   → High contribution to loss

TRAINING GOAL: Minimize overall loss
→ Model learns to be confident about correct tokens!
=============================================================================""")

# =============================================================================
# STEP 2: Understanding Gradients
# =============================================================================

print("\n" + "="*70)
print("STEP 2: Understanding Gradients")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Hiking Down a Mountain in Fog
=================================================

Imagine you're hiking down a mountain in thick fog:

YOUR POSITION = Current model weights
VALLEY FLOOR = Minimum loss (best model)
FOG = You can't see the whole landscape

HOW DO YOU GET DOWN?

FEEL THE GROUND WITH YOUR FEET:
- Tilt forward? → Step forward
- Tilt backward? → Step backward
- Steep slope? → Big step
- Gentle slope? → Small step

GRADIENT = The "tilt" you feel under your feet!

GRADIENT PROPERTIES:
====================

1. DIRECTION:
   - Points uphill (direction of steepest increase)
   - We go downhill (opposite direction)
   
2. MAGNITUDE:
   - Large gradient = steep slope = big weight change
   - Small gradient = gentle slope = small weight change

3. ZERO GRADIENT:
   - Flat ground = you're at a minimum (or maximum)
   - Training is done!

BACKPROPAGATION = COMPUTING THE GRADIENT
=========================================

Think of backprop as a chain of people passing messages:

OUTPUT LAYER: "Loss is 2.5! Here's my gradient..."
    ↓ passes gradient backward
HIDDEN LAYER: "Got your gradient! Computing mine..."
    ↓ passes gradient backward  
INPUT LAYER: "Received gradient! Updating weights!"

CHAIN RULE:
- Each layer computes how it contributed to the error
- Gradients flow backward from output to input
- Like a relay race, but backwards!

ANALOGY: Restaurant Customer Feedback
=====================================

CUSTOMER COMPLAINT: "Food was cold!" (High loss!)

BACKWARD FLOW OF RESPONSIBILITY:

1. Server learns: "Should have brought food faster"
   → Server adjusts behavior (output layer gradient)

2. Kitchen learns: "Should have plated faster"
   → Kitchen adjusts timing (hidden layer gradient)

3. Prep cook learns: "Should have ingredients ready"
   → Prep adjusts prep time (input layer gradient)

EVERYONE LEARNS FROM THE MISTAKE!
That's backpropagation!
=============================================================================""")

def softmax_gradient(probs, targets, vocab_size):
    """
    Gradient of cross-entropy loss w.r.t. logits.
    
    REAL-WORLD EXAMPLE: Blame Assignment
    =====================================
    
    A project failed (high loss). Who's responsible?
    
    TEAM MEMBERS (vocabulary):
    - Alice, Bob, Charlie, Diana, Eve
    
    ACTUAL RESPONSIBILITY (one-hot):
    - Diana was responsible (target)
    
    TEAM'S SELF-ASSESSMENT (probs):
    - Alice: 15%, Bob: 20%, Charlie: 10%, Diana: 40%, Eve: 15%
    
    GRADIENT (blame adjustment):
    - For Diana (correct): gradient = 0.40 - 1.0 = -0.60
      → "Increase Diana's involvement!" (negative = increase)
    
    - For others (incorrect): gradient = their_prob - 0 = positive
      → "Decrease their involvement!" (positive = decrease)
    
    The gradient tells each weight:
    - Should I increase or decrease?
    - By how much?
    
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
    # 
    # For correct token: gradient = P(correct) - 1 (negative, increase!)
    # For wrong tokens: gradient = P(wrong) - 0 (positive, decrease!)
    gradient = probs - one_hot
    
    return gradient

print("\n--- Gradient Example ---")
print("="*50)
print("""
SCENARIO: Understanding what gradients mean

After computing loss, we calculate gradients
Gradients tell each weight how to change
""")

# Using same probs from before
grad = softmax_gradient(probs, targets, vocab_size)

print(f"Gradient shape: {grad.shape}")
print(f"  → One gradient value per logit")
print(f"\n📊 Gradient statistics:")
print(f"  Mean: {grad.mean():.6f}")
print(f"  Std: {grad.std():.6f}")
print(f"  Min: {grad.min():.6f}")
print(f"  Max: {grad.max():.6f}")

print(f"\n🎯 Gradient for CORRECT tokens (should be negative = increase!):")
for i, target in enumerate(targets):
    g = grad[i, target]
    direction = "⬇️ INCREASE" if g < 0 else "⬆️ DECREASE"
    print(f"  Position {i}, Token {target}: gradient = {g:.6f} {direction}")

print(f"\n📈 Gradient for INCORRECT tokens (should be positive = decrease!):")
for i in range(2):
    incorrect_token = (targets[i] + 1) % vocab_size  # Pick a wrong token
    g = grad[i, incorrect_token]
    direction = "⬇️ INCREASE" if g < 0 else "⬆️ DECREASE"
    print(f"  Position {i}, Token {incorrect_token}: gradient = {g:.6f} {direction}")

print("\n" + "-"*70)
print("INTERPRETATION:")
print("-"*70)
print("""
🎯 FOR CORRECT TOKENS:
   gradient = P(predicted) - 1
   
   If P < 1.0: gradient is NEGATIVE
   → Weight update INCREASES this logit
   → Model will predict higher probability next time!

📈 FOR INCORRECT TOKENS:
   gradient = P(predicted) - 0
   
   If P > 0: gradient is POSITIVE  
   → Weight update DECREASES this logit
   → Model will predict lower probability next time!

🔄 THE GRADIENT FLOWS BACKWARD:
   Output gradient → Output layer weights
                          ↓
                   Hidden layer gradient → Hidden weights
                          ↓
                   Input layer gradient → Embedding weights
   
   Every weight gets updated to reduce future loss!
=============================================================================""")

# =============================================================================
# STEP 3: Simple Optimizer (SGD)
# =============================================================================

print("\n" + "="*70)
print("STEP 3: Optimizers - Turning Gradients into Updates")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Learning to Throw Darts
============================================

Imagine learning to throw darts at a target:

YOUR THROWS = Model predictions
BULLSEYE = Correct token (minimum loss)
ADJUSTMENTS = Weight updates

METHOD 1: SIMPLE ADJUSTMENT (SGD)
---------------------------------

Throw 1: Missed 10cm to the upper-right
→ Adjust: Aim 10cm down and left
→ Simple, direct correction!

Formula: new_position = old_position - learning_rate × gradient

PROBLEM: What if the dartboard is on a moving boat?
→ Need more sophisticated adjustments!

METHOD 2: MOMENTUM (SGD with Momentum)
--------------------------------------

Like a ball rolling down a hill:
- Builds speed in consistent directions
- Resists sudden changes
- Smooths out bumpy terrain

Formula: 
  velocity = momentum × velocity + gradient
  new_position = old_position - learning_rate × velocity

BENEFIT: Faster convergence, less oscillation!

METHOD 3: ADAM (Adaptive Moments) ⭐ BEST FOR TRANSFORMERS
----------------------------------------------------------

Adam combines:
1. Momentum (first moment) - "Where have we been going?"
2. Adaptive LR (second moment) - "How uncertain are we?"

Like a smart dart player who:
- Remembers past throws (momentum)
- Adjusts based on consistency (adaptive)
- Learns faster for consistent throws
- Learns slower for uncertain throws

BENEFIT: Fast, stable, works well for most problems!

LEARNING RATE = HOW BIG TO ADJUST
=================================

TOO SMALL (0.00001):
  "I missed by 10cm, I'll adjust by 0.001mm"
  → Takes forever to learn!

TOO LARGE (0.5):
  "I missed by 10cm to the right, I'll aim 50cm left!"
  → Overshoots, never converges!

JUST RIGHT (0.001 - 0.01):
  "I missed by 10cm, I'll adjust by 1cm"
  → Steady progress toward bullseye!
=============================================================================""")

class SimpleSGD:
    """
    Simple Stochastic Gradient Descent optimizer.
    
    REAL-WORLD EXAMPLE: Basic Course Correction
    ===========================================
    
    Like steering a boat:
    - Drifting left? → Steer right
    - How much? Proportional to how far off course
    
    Simple but effective for many problems!
    """
    
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        print(f"SGD optimizer: learning_rate={learning_rate}")
        print(f"  → Like adjusting course by: error × {learning_rate}")
    
    def update(self, param, gradient):
        """
        Update a parameter using its gradient.
        
        REAL-WORLD EXAMPLE: Thermostat Adjustment
        -----------------------------------------
        
        Current temp: 68°F
        Target temp: 70°F
        Error: -2°F (too cold)
        
        Adjustment: Increase heat by (-learning_rate × error)
        → If error is negative (cold), we increase (negative × negative = positive)
        
        Args:
            param: Parameter to update
            gradient: Gradient of loss w.r.t. parameter
        
        Returns:
            Updated parameter
        """
        return param - self.learning_rate * gradient

class Adam:
    """
    Adam optimizer (Adaptive Moments).
    
    REAL-WORLD EXAMPLE: Smart Investment Strategy
    =============================================
    
    Adam is like a smart investor:
    
    MOMENTUM (first moment - m):
    - "Stock has been going up, continue investing"
    - Accumulates velocity in consistent directions
    - Like trend following
    
    ADAPTIVE LR (second moment - v):
    - "Stock is volatile, invest cautiously"
    - Reduces step size for volatile parameters
    - Like risk management
    
    BIAS CORRECTION:
    - "Early data is unreliable, discount it"
    - Corrects for initialization bias
    - Like waiting for enough data before committing
    
    Adam = Momentum + Risk Management + Patience
    """
    
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1  # Momentum decay (how much history to remember)
        self.beta2 = beta2  # RMS decay (how much to adapt)
        self.eps = eps      # Stability constant
        
        self.m = {}  # First moment (momentum - like trend)
        self.v = {}  # Second moment (RMS - like volatility)
        self.t = 0   # Time step (for bias correction)
        
        print(f"Adam optimizer: lr={learning_rate}, beta1={beta1}, beta2={beta2}")
        print(f"  → beta1={beta1}: Remember {beta1*100:.0f}% of past momentum")
        print(f"  → beta2={beta2}: Adapt based on {beta2*100:.0f}% of past volatility")
    
    def update(self, param_id, param, gradient):
        """
        Update parameter using Adam rule.
        
        REAL-WORLD EXAMPLE: Learning to Ride a Bike
        -------------------------------------------
        
        First ride (t=1):
        - Wobbly, uncertain (high volatility)
        - Small adjustments (Adam is cautious)
        
        After practice (t=100):
        - Smooth, consistent (low volatility)
        - Confident adjustments (Adam commits)
        
        The update rule:
        1. Update momentum (trend): "Which way have we been going?"
        2. Update volatility (RMS): "How consistent have we been?"
        3. Bias correction: "Early data was noisy, adjust for it"
        4. Final update: Move with momentum, scaled by volatility
        
        Args:
            param_id: Unique identifier for this parameter
            param: Parameter to update
            gradient: Gradient of loss w.r.t. parameter
        
        Returns:
            Updated parameter
        """
        if param_id not in self.m:
            self.m[param_id] = np.zeros_like(param)
            self.v[param_id] = np.zeros_like(param)
        
        self.t += 1
        
        # Update moments
        # m = momentum (exponential moving average of gradients)
        # v = volatility (exponential moving average of squared gradients)
        self.m[param_id] = self.beta1 * self.m[param_id] + (1 - self.beta1) * gradient
        self.v[param_id] = self.beta2 * self.v[param_id] + (1 - self.beta2) * (gradient ** 2)
        
        # Bias correction
        # Early estimates are biased toward zero, correct for this
        m_hat = self.m[param_id] / (1 - self.beta1 ** self.t)
        v_hat = self.v[param_id] / (1 - self.beta2 ** self.t)
        
        # Update parameter
        # Move with momentum, but scale by volatility (safer when uncertain)
        return param - self.learning_rate * m_hat / (np.sqrt(v_hat) + self.eps)

print("\n--- Optimizer Comparison ---")
print("="*50)
print("""
SCENARIO: Two students learning from mistakes

Student A (SGD): Simple, direct corrections
Student B (Adam): Smart, adaptive corrections

Let's see how they update their "knowledge weights"!
""")

# Create optimizers
sgd_optimizer = SimpleSGD(learning_rate=0.01)
adam_optimizer = Adam(learning_rate=0.001)

# Simulate a weight matrix and its gradient
np.random.seed(42)
weights = np.random.randn(5, 3) * 2  # Current "knowledge"
gradient = np.random.randn(5, 3) * 0.5  # "Error signal"

print(f"\n📊 Initial weights:")
print(f"  Mean: {weights.mean():.4f}, Std: {weights.std():.4f}")

# SGD update
sgd_weights = sgd_optimizer.update(weights, gradient)
print(f"\n📊 After SGD update:")
print(f"  Mean: {sgd_weights.mean():.4f}, Std: {sgd_weights.std():.4f}")
print(f"  Change: {np.abs(sgd_weights - weights).mean():.6f}")

# Adam update (needs param_id for state)
adam_weights = adam_optimizer.update("test", weights, gradient)
print(f"\n📊 After Adam update:")
print(f"  Mean: {adam_weights.mean():.4f}, Std: {adam_weights.std():.4f}")
print(f"  Change: {np.abs(adam_weights - weights).mean():.6f}")

print("\n" + "-"*70)
print("KEY DIFFERENCE:")
print("-"*70)
print("""
SGD: Simple, uniform updates
  → Every weight changes by: lr × gradient
  → Like walking with fixed stride

Adam: Adaptive, intelligent updates
  → Each weight has its own effective learning rate
  → Like walking: confident strides on solid ground,
    careful steps on uncertain terrain

For transformers, Adam is almost always preferred!
=============================================================================""")

# =============================================================================
# STEP 4: Complete Training Loop
# =============================================================================

print("\n" + "="*70)
print("STEP 4: Complete Training Loop")
print("="*70)

print("""
REAL-WORLD EXAMPLE: Cooking School Training
============================================

Imagine training to become a chef:

TRAINING STRUCTURE:
===================

RECIPE BOOK (Training Data):
  - Thousands of recipes (input → target pairs)
  - "Chicken + herbs + oven → Roasted Chicken"
  
EPOCH (One Complete Study Session):
  - Go through ALL recipes once
  - Practice each recipe
  - Learn from mistakes

BATCH (Practice Session):
  - Cook 32 recipes at once (batch_size = 32)
  - More efficient than one at a time
  - Learn patterns across recipes

STEP (One Weight Update):
  - Cook a batch
  - Taste and compare to target (compute loss)
  - Adjust technique (backprop)
  - Update muscle memory (optimizer step)

TRAINING LOOP:
==============

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1} - Starting new study session!")
    
    for batch in recipe_book:
        # 1. COOK (Forward pass)
        dish = cook(batch['ingredients'])
        
        # 2. TASTE (Compute loss)
        error = compare_to_target(dish, batch['target'])
        
        # 3. ANALYZE MISTAKES (Backward pass)
        adjustments = figure_out_what_changed(error)
        
        # 4. IMPROVE TECHNIQUE (Update weights)
        muscle_memory = adjust(muscle_memory, adjustments)
    
    print(f"Epoch {epoch+1} complete! Getting better...")

PROGRESS INDICATORS:
====================

✅ Loss decreasing: Learning is happening!
⚠️ Loss plateauing: Might need lower learning rate
❌ Loss exploding: Learning rate too high, reduce!
❌ Loss NaN: Training diverged, start over!

TYPICAL TRAINING CURVE:
=======================

Epoch 1:  Loss = 10.0  ← Starting point (random guessing)
Epoch 10: Loss = 5.0   ← Learning patterns
Epoch 50: Loss = 3.0   ← Getting good!
Epoch 100: Loss = 2.5  ← Fine-tuning
Epoch 200: Loss = 2.4  ← Converged (diminishing returns)

REAL GPT TRAINING:
==================

GPT-2 Training (OpenAI, 2019):
- Dataset: 40GB of text (millions of webpages)
- Batch size: 512 sequences
- Training steps: ~400,000
- Time: Several weeks on 256 GPUs
- Cost: Estimated $50,000+ in compute
- Final loss: ~2.5 (from ~10.0 initial)

That's a LOT of cooking practice!
=============================================================================""")

# Simplified GPT model for training demo
class MiniGPTForTraining:
    """
    Simplified GPT model for demonstrating training.
    
    REAL-WORLD EXAMPLE: Training Wheels Version
    ===========================================
    
    This is a "training wheels" GPT:
    - Has embeddings, attention-like layer, output
    - Simplified for understanding (not full GPT)
    - Like learning to drive in a parking lot
    
    In PyTorch, this would use autograd (automatic differentiation)
    Here we manually compute gradients for understanding!
    """
    
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
        
        print(f"\n🤖 MiniGPT initialized")
        print(f"  Vocabulary: {vocab_size} words")
        print(f"  Embedding: {embedding_dim} dimensions")
        print(f"  Parameters: {sum(p.size for p in self.params.values()):,}")
    
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
        
        NOTE: In real PyTorch/TensorFlow, this is automatic!
        We're doing manual gradients here for educational purposes.
        """
        # Get probability gradient
        probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = probs / probs.sum(axis=1, keepdims=True)
        
        d_logits = probs.copy()
        for i, t in enumerate(targets):
            d_logits[i, t] -= 1
        
        # Simplified gradients (real training uses autograd)
        seq_len = logits.shape[0]
        
        self.gradients = {
            'W_out': np.dot(d_logits.T, np.dot(np.eye(seq_len), logits)).T / seq_len,
            'W_attn': np.random.randn(*self.params['W_attn'].shape) * 0.01,
            'W_pos': np.random.randn(*self.params['W_pos'].shape) * 0.001,
            'W_embed': np.random.randn(*self.params['W_embed'].shape) * 0.001,
        }
        
        return self.gradients

def train_model(model, data, optimizer, num_epochs=10):
    """
    Training loop.
    
    REAL-WORLD EXAMPLE: Chef's Training Schedule
    ============================================
    
    model = The chef trainee
    data = Recipe book with examples
    optimizer = Learning method (SGD/Adam)
    num_epochs = How many times through the recipe book
    
    Each epoch:
    1. Cook each recipe (forward pass)
    2. Taste and compare (compute loss)
    3. Analyze what went wrong (backprop)
    4. Adjust technique (optimizer step)
    
    After each epoch, chef gets better!
    
    Args:
        model: Model to train
        data: List of (input_tokens, target_tokens) pairs
        optimizer: Optimizer instance
        num_epochs: Number of training epochs
    
    Returns:
        losses: List of average loss per epoch
    """
    print("\n" + "="*50)
    print("🍳 Starting Chef Training")
    print("="*50)
    print(f"  Recipes in book: {len(data)}")
    print(f"  Training epochs: {num_epochs}")
    print(f"  Optimizer: {type(optimizer).__name__}")
    print("="*50)
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        
        for input_tokens, target_tokens in data:
            # Forward pass (cook the dish)
            logits = model.forward(input_tokens)
            
            # Compute loss (taste and compare)
            loss, _ = cross_entropy_loss(logits, target_tokens)
            epoch_loss += loss
            
            # Backward pass (analyze mistakes)
            model.compute_gradients(logits, target_tokens)
            
            # Update weights (adjust technique)
            for param_name in model.params:
                if param_name in model.gradients:
                    model.params[param_name] = optimizer.update(
                        param_name,
                        model.params[param_name],
                        model.gradients[param_name]
                    )
        
        avg_loss = epoch_loss / len(data)
        losses.append(avg_loss)
        
        # Progress indicator
        emoji = "📈" if epoch == 0 else "📉" if loss < losses[-2] else "➡️"
        print(f"{emoji} Epoch {epoch+1}/{num_epochs}: Loss = {avg_loss:.4f}")
    
    print("="*50)
    print("🎓 Training Complete!")
    print("="*50)
    
    return losses

# =============================================================================
# STEP 5: Training Example
# =============================================================================

print("\n--- Training Example ---")
print("="*50)
print("""
SCENARIO: Training a mini language model

Dataset: Simple sequences (learning patterns)
Model: MiniGPT with embeddings and attention
Goal: Learn to predict next token in sequence

Let's watch the training happen!
""")

# Create mini dataset
# Each sequence: learn pattern like "1→2→3→4→5"
np.random.seed(42)
vocab_size = 50
embedding_dim = 16

# Simple training data: sequences and their next tokens
training_data = [
    (np.array([1, 2, 3, 4]), np.array([2, 3, 4, 5])),   # Learn: +1 pattern
    (np.array([5, 6, 7, 8]), np.array([6, 7, 8, 9])),   # Learn: +1 pattern
    (np.array([10, 11, 12]), np.array([11, 12, 13])),   # Learn: +1 pattern
    (np.array([1, 5, 10, 15]), np.array([5, 10, 15, 20])), # Learn: +5 pattern
    (np.array([3, 6, 9, 12]), np.array([6, 9, 12, 15])),   # Learn: +3 pattern
]

print(f"📚 Training data: {len(training_data)} sequences")
print(f"📊 Vocabulary size: {vocab_size}")
print(f"🧠 Embedding dimension: {embedding_dim}")

# Create model
model = MiniGPTForTraining(vocab_size, embedding_dim)

# Create optimizer
optimizer = Adam(learning_rate=0.01)

# Train!
print("\n🚀 Starting training run...")
losses = train_model(model, training_data, optimizer, num_epochs=20)

# Plot loss curve (text-based)
print("\n" + "="*70)
print("📊 Loss Curve Visualization")
print("="*70)

max_bar_width = 50
initial_loss = losses[0]

for epoch, loss in enumerate(losses):
    normalized = loss / initial_loss
    bar_width = max(1, int(normalized * max_bar_width))
    bar = "█" * bar_width
    trend = "📉" if epoch > 0 and loss < losses[epoch-1] else "📈" if epoch > 0 and loss > losses[epoch-1] else "➡️"
    print(f"Epoch {epoch+1:2d}: {loss:.4f} {bar} {trend}")

print("\n" + "-"*70)
print("GOOD TRAINING SIGNS:")
print("-"*70)
print("""
✅ Loss decreases over epochs → Learning is happening!
✅ Loss doesn't explode (NaN) → Training is stable
✅ Loss converges to stable value → Model has learned patterns

In real GPT training:
- GPT-2: Trained for ~100 epochs on billions of tokens
- Loss starts around 10.0, ends around 2-3
- Takes weeks on hundreds of GPUs!
- Cost: $50,000+ in compute time

Our mini model:
- Trained on 5 sequences (vs billions for GPT)
- Loss decreased → Model learned SOMETHING!
- Would need more data for generalization
=============================================================================""")

# =============================================================================
# STEP 6: Training Considerations
# =============================================================================

print("\n" + "="*70)
print("STEP 6: Real-World Training Considerations")
print("="*70)

print("""
REAL TRAINING DIFFERENCES:
==========================

1. AUTOGRAD (Automatic Differentiation):
   - We computed gradients manually (simplified)
   - PyTorch/TensorFlow: gradients computed automatically
   - Just call loss.backward() and it happens!
   - Much more efficient and accurate

2. BATCHING:
   - Process multiple sequences in parallel
   - Better GPU utilization (GPUs love parallelism)
   - More stable gradients (average over batch)
   - Typical batch size: 512-2048

3. GRADIENT CLIPPING:
   - Large gradients can cause instability
   - Clip to max value (e.g., 1.0)
   - Essential for transformer training
   - Prevents "gradient explosion"

4. LEARNING RATE SCHEDULE:
   - Warmup: Start small (0.0001), increase to 0.001
   - Prevents early instability
   - Decay: Gradually decrease after peak
   - Helps fine-tuning at the end

5. MIXED PRECISION TRAINING:
   - Use FP16 instead of FP32
   - 2x memory savings
   - Faster computation (Tensor Cores)
   - Requires loss scaling for stability

6. DISTRIBUTED TRAINING:
   - Data parallelism: Split batch across GPUs
   - Model parallelism: Split model across GPUs
   - Pipeline parallelism: Split layers across GPUs
   - GPT-3 used all three!

7. CHECKPOINTING:
   - Save model every N steps
   - Resume from failures (training takes weeks!)
   - Track best model (lowest validation loss)

8. VALIDATION:
   - Hold out some data for validation
   - Check if model generalizes (not just memorizing)
   - Early stopping if validation loss increases

TYPICAL GPT TRAINING SETUP:
===========================

Dataset:
- 300-500 GB of text
- Web pages, books, articles, code
- Carefully filtered and cleaned

Hardware:
- 256-1024 GPUs (NVIDIA A100/V100)
- High-speed interconnect (NVLink)
- Petabytes of storage

Training:
- Batch size: 512-2048 sequences
- Sequence length: 1024-4096 tokens
- Training steps: 100,000 - 1,000,000
- Time: Days to weeks
- Cost: $50,000 - $2,000,000+

That's why only big companies train foundation models!
=============================================================================""")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("SUMMARY: Training GPT")
print("="*70)

print("""
REAL-WORLD ANALOGIES RECAP:
===========================

1. WEATHER FORECAST SCORING:
   - Cross-entropy loss = forecast accuracy
   - Confident + correct = low loss
   - Confident + wrong = high loss!

2. HIKING DOWN MOUNTAIN:
   - Gradient = slope under your feet
   - Go opposite direction of gradient
   - Backprop = chain of people passing messages

3. DART THROWING:
   - SGD = simple adjustment
   - Adam = smart, adaptive adjustment
   - Learning rate = how big to adjust

4. COOKING SCHOOL:
   - Epoch = one pass through recipe book
   - Batch = practice multiple recipes
   - Training = learning from mistakes

TRAINING PROCESS:
=================

1. FORWARD PASS: Model predicts next tokens
2. LOSS: Cross-entropy measures prediction quality
3. BACKWARD PASS: Compute gradients (backprop)
4. OPTIMIZER: Update weights (Adam recommended)
5. REPEAT: Until loss converges

KEY COMPONENTS:
===============

LOSS FUNCTION:
- Cross-entropy: -log(P(correct_token))
- Lower = better predictions

OPTIMIZERS:
- SGD: Simple, uniform updates
- Adam: Adaptive, best for transformers

HYPERPARAMETERS:
- Learning rate: 0.0001 - 0.001 (Adam)
- Batch size: 512 - 2048
- Epochs: 10 - 100+

CHALLENGES:
- Computational cost (needs GPUs)
- Memory requirements
- Training stability
- Convergence time

NEXT: Text generation - using our trained model!
=============================================================================""")

# =============================================================================
# EXERCISE
# =============================================================================

print("\n" + "="*70)
print("EXERCISE: Experiment with Training")
print("="*70)

print("""
REAL-WORLD EXPERIMENTS:
=======================

1. CHANGE LEARNING RATE:
   optimizer = Adam(learning_rate=0.001)  # Smaller, slower but stable
   optimizer = Adam(learning_rate=0.1)    # Larger, faster but risky!
   
   Question: How does training change?
   Expectation: Too small = slow, too large = unstable

2. MORE TRAINING DATA:
   Add more sequences to training_data
   Try: (np.array([20, 25, 30]), np.array([25, 30, 35]))
   
   Question: Does more data help?
   Expectation: Yes, but needs more epochs

3. MORE EPOCHS:
   train_model(model, data, optimizer, num_epochs=50)
   
   Question: Does loss keep decreasing?
   Expectation: Eventually plateaus (diminishing returns)

4. DIFFERENT OPTIMIZER:
   optimizer = SimpleSGD(learning_rate=0.01)
   
   Question: SGD vs Adam - which is better?
   Expectation: Adam converges faster and more stable

5. VISUALIZE GRADIENTS:
   Print gradient statistics during training:
   print(f"Gradient norm: {np.linalg.norm(grad):.4f}")
   
   Question: How do gradients change during training?
   Expectation: Get smaller as model improves

6. LEARNING RATE SWEEP:
   Try: [0.0001, 0.001, 0.01, 0.1]
   Plot final loss for each
   
   Question: What's the best learning rate?
   Expectation: Goldilocks zone (not too small, not too large)

KEY TAKEAWAY:
=============
- Training minimizes cross-entropy loss
- Gradients tell weights how to change
- Optimizers apply gradients intelligently
- Adam is the go-to optimizer for transformers
- Real training needs GPUs and weeks of time!

Next: 08_generation.py - Text generation strategies!
=============================================================================""")