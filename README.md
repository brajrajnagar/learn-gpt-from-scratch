# Learn GPT From Scratch - Build Your Own Transformer

A complete step-by-step course that teaches you how GPT works by building one from scratch using Python and NumPy.

## 🎯 What You'll Build

By the end of this course, you'll have built a **working Mini GPT** that can:
- Understand the complete transformer architecture
- Process text and generate predictions
- Train on text data (forward pass demonstration)
- Generate text with multiple strategies (greedy, sampling, top-k, top-p)

## 📚 Course Structure

### Part 1: Foundations

| Lesson | File | What You Learn |
|--------|------|----------------|
| 1 | `01_neural_network_basics.py` | Single neuron → dense layers → forward propagation |
| 2 | `02_embeddings.py` | How words become vectors (token + position embeddings) |
| 3 | `03_attention.py` | Self-attention mechanism (Q, K, V) |
| 4 | `04_multihead_attention.py` | Multiple attention heads working in parallel |

### Part 2: Building the Transformer

| Lesson | File | What You Learn |
|--------|------|----------------|
| 5 | `05_transformer_block.py` | Complete transformer block (attention + FFN + residuals) |
| 6 | `06_gpt_model.py` | Full GPT architecture (stacked blocks + output projection) |

### Part 3: Training and Generation

| Lesson | File | What You Learn |
|--------|------|----------------|
| 7 | `07_training.py` | Cross-entropy loss, training loop, perplexity |
| 8 | `08_generation.py` | Greedy decoding, sampling, top-k, top-p, temperature |
| 9 | `09_mini_gpt.py` | **Complete working Mini GPT** - everything together! |

## 🚀 Quick Start

### Prerequisites

```bash
# You only need NumPy!
pip install numpy
```

### Run the Course

```bash
# Start from the beginning
python 01_neural_network_basics.py

# Work through each lesson sequentially
python 02_embeddings.py
python 03_attention.py
python 04_multihead_attention.py
python 05_transformer_block.py
python 06_gpt_model.py
python 07_training.py
python 08_generation.py

# Finally, run the complete Mini GPT!
python 09_mini_gpt.py
```

## 📖 What Each Lesson Covers

### Lesson 1: Neural Network Basics
**Real-World Analogy:** Restaurant Order System

Learn how a single neuron detects patterns, then combine neurons into layers. By the end, you'll understand:
- Dot products (pattern matching)
- Bias (baseline adjustment)
- ReLU activation (firing threshold)
- Softmax (probability distribution)

### Lesson 2: Embeddings
**Real-World Analogy:** Library Card Catalog

Learn how words become mathematical vectors:
- Token embeddings (word meaning)
- Position embeddings (word order)
- Combined representation

### Lesson 3: Self-Attention
**Real-World Analogy:** Team of Detectives

Learn the core innovation of transformers:
- Query (what am I looking for?)
- Key (what do I contain?)
- Value (what information do I carry?)
- Attention scores (relevance weighting)

### Lesson 4: Multi-Head Attention
**Real-World Analogy:** Specialist Analysis Team

Learn how multiple attention heads work in parallel:
- Split embeddings into heads
- Each head focuses on different aspects
- Combine results for rich understanding

### Lesson 5: Transformer Block
**Real-World Analogy:** Document Processing Pipeline

Learn the complete transformer block:
- Layer normalization (stability)
- Feed-forward network (transformation)
- Residual connections (information flow)

### Lesson 6: Complete GPT Model
**Real-World Analogy:** Restaurant Assembly Line

Learn the full GPT architecture:
- Input embeddings
- Stacked transformer blocks
- Output projection
- Probability distribution over vocabulary

### Lesson 7: Training
**Real-World Analogy:** Student Learning Process

Learn how GPT learns from data:
- Cross-entropy loss (measuring errors)
- Training loop (iterative improvement)
- Perplexity (confidence metric)

### Lesson 8: Text Generation
**Real-World Analogy:** Creative Writing Strategies

Learn different generation strategies:
- Greedy decoding (always pick best)
- Sampling (pick randomly from distribution)
- Top-k sampling (pick from best k)
- Top-p sampling (pick until probability threshold)
- Temperature (control creativity)

### Lesson 9: Mini GPT
**Real-World Implementation:** Complete working model!

Everything comes together in a working Mini GPT that:
- Tokenizes text (character-level)
- Processes through transformer blocks
- Generates text with multiple strategies

## 🎓 Key Concepts Explained

### How GPT Works (Simple Explanation)

```
Input: "The cat sat on the"
       │
       ▼
1. Convert words to vectors (embeddings)
       │
       ▼
2. Add position information (order matters!)
       │
       ▼
3. Process through transformer blocks (understanding)
       │
       ▼
4. Get probabilities for all words (prediction)
       │
       ▼
5. Sample next word: "mat" ← most likely!
```

### GPT Architecture Comparison

| Component | GPT-2 Small | Our Mini GPT |
|-----------|-------------|--------------|
| Parameters | 124M | ~100K |
| Embedding | 768 | 64 |
| Heads | 12 | 4 |
| Blocks | 12 | 2 |
| Vocabulary | 50,257 | ~30 (chars) |
| Training Data | 40GB | Demo text |

**Same architecture, different scale!**

## 🔧 Experiment Guide

### Try These Experiments

1. **Change Model Size**
   ```python
   # In 09_mini_gpt.py, modify:
   model = MiniGPT(
       vocab_size=tokenizer.vocab_size,
       max_seq_len=64,
       dim=128,        # Try: 32, 64, 128, 256
       num_heads=8,    # Try: 2, 4, 8
       num_blocks=4,   # Try: 1, 2, 4
       ff_dim=512
   )
   ```

2. **Try Different Generation Strategies**
   ```python
   # Greedy (focused)
   model.generate(tokens, temperature=0.5)
   
   # Creative (diverse)
   model.generate(tokens, temperature=1.5)
   
   # Balanced (recommended)
   model.generate(tokens, top_p=0.9, temperature=1.0)
   ```

3. **Train on Your Own Text**
   ```python
   # Replace TRAINING_TEXT in 09_mini_gpt.py
   TRAINING_TEXT = """
   Your text here!
   The more data, the better!
   """
   ```

## 📚 Additional Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 details

### Blogs
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's minimal GPT

### Next Steps
1. **Learn PyTorch** - Real deep learning framework
2. **Study Backpropagation** - How models actually learn
3. **Explore BPE Tokenization** - How GPT-2/3 tokenize text
4. **Try nanoGPT** - Minimal PyTorch GPT implementation

## 🎉 What You've Accomplished

After completing this course:

✅ You understand how GPT works at a fundamental level
✅ You can explain self-attention, embeddings, and transformers
✅ You've built a working language model from scratch
✅ You understand generation strategies (greedy, sampling, top-k, top-p)
✅ You're ready to dive into more advanced deep learning!

## ⚠️ Important Notes

This is an **educational implementation**:
- Uses NumPy (not PyTorch/TensorFlow)
- Demonstrates concepts (not optimized for performance)
- Forward pass only (no backpropagation)
- Character-level tokenizer (not BPE)

For production use, see:
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal PyTorch GPT
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Production library

---

**Happy Learning! 🚀**

Start with `python 01_neural_network_basics.py` and work through each lesson!