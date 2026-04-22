# Learn GPT From Scratch - Build Your Own Transformer

This project will teach you how GPT (Generative Pre-trained Transformer) works by building one from scratch using Python and NumPy.

## 📚 Learning Path

### Part 1: Foundations
1. **Neural Networks Basics** - Understanding neurons, layers, and forward propagation
2. **Embeddings** - How words become numbers
3. **Attention Mechanism** - The core of transformers
4. **Positional Encoding** - Giving order to sequences

### Part 2: Building GPT
5. **Multi-Head Attention** - The heart of transformer
6. **Feed Forward Networks** - Processing attention output
7. **Layer Normalization** - Stabilizing training
8. **Complete GPT Block** - Combining all components

### Part 3: Training & Generation
9. **Data Preparation** - Tokenization and datasets
10. **Training Loop** - Loss functions and optimization
11. **Text Generation** - Sampling strategies
12. **Your Own GPT** - A working mini language model

## 🛠️ Setup & Installation

### First Time Setup

1. **Create and activate virtual environment** (already set up):
   ```bash
   # macOS/Linux
   source venv/bin/activate

   # Windows
   venv\Scripts\activate
   ```

2. **Install dependencies** (already installed):
   ```bash
   pip install numpy
   ```

### Running the Lessons

Once the virtual environment is activated:

```bash
# Start with lesson 1
python 01_neural_network_basics.py

# Then proceed through each lesson
python 02_embeddings.py
python 03_attention.py
# ... and so on

# Finally, run the complete mini GPT
python 09_mini_gpt.py
```

## 📁 Project Structure

```
gpt/
├── README.md                      # This file
├── venv/                          # Virtual environment (auto-created)
├── 01_neural_network_basics.py    - Build a simple neural network
├── 02_embeddings.py               - Learn word embeddings
├── 03_attention.py                - Understand attention mechanism
├── 04_multihead_attention.py      - Multi-head attention implementation
├── 05_transformer_block.py        - Complete transformer block
├── 06_gpt_model.py                - Full GPT model
├── 07_training.py                 - Training the model
├── 08_generation.py               - Text generation
├── 09_mini_gpt.py                 - Complete mini GPT implementation
└── data/                          - Training data
```

## 🚀 Getting Started

1. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

2. **Start with lesson 1:**
   ```bash
   python 01_neural_network_basics.py
   ```

3. **Work through each file sequentially** - each lesson builds on the previous concepts!

## 📝 Notes

- All lessons use only NumPy - no heavy frameworks needed!
- The virtual environment keeps dependencies isolated from your global Python installation
- When you're done learning, deactivate with: `deactivate`

Let's begin!