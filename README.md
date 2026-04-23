# Learn GPT From Scratch - Build Your Own Transformer

A complete step-by-step course that teaches you how GPT works by building one from scratch using Python and NumPy, culminating in a PyTorch implementation with real backpropagation!

## 🎯 What You'll Build

By the end of this course, you'll have built:
1. **Mini GPT (NumPy)** - Educational implementation understanding every detail
2. **Mini GPT (PyTorch)** - Production-ready code with real training!

## 📚 Course Structure

### Part 1: Foundations (NumPy)

| Lesson | File | What You Learn |
|--------|------|----------------|
| 1 | `01_neural_network_basics.py` | Single neuron → dense layers → forward propagation |
| 2 | `02_embeddings.py` | How words become vectors (token + position embeddings) |
| 3 | `03_attention.py` | Self-attention mechanism (Q, K, V) |
| 4 | `04_multihead_attention.py` | Multiple attention heads working in parallel |

### Part 2: Building the Transformer (NumPy)

| Lesson | File | What You Learn |
|--------|------|----------------|
| 5 | `05_transformer_block.py` | Complete transformer block (attention + FFN + residuals) |
| 6 | `06_gpt_model.py` | Full GPT architecture (stacked blocks + output projection) |

### Part 3: Training and Generation (NumPy)

| Lesson | File | What You Learn |
|--------|------|----------------|
| 7 | `07_training.py` | Cross-entropy loss, training loop, perplexity |
| 8 | `08_generation.py` | Greedy decoding, sampling, top-k, top-p, temperature |
| 9 | `09_mini_gpt.py` | **Complete working Mini GPT** - everything together! |

### Part 4: Production Implementation (PyTorch) ⭐

| Lesson | File | What You Learn |
|--------|------|----------------|
| 10 | `10_pytorch_transformer.py` | PyTorch implementation with **real backpropagation**! |

## 🚀 Quick Start

### Prerequisites

```bash
# For Lessons 1-9 (NumPy only)
pip install numpy

# For Lesson 10 (PyTorch - real training!)
pip install torch
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
python 09_mini_gpt.py

# Finally, see the PyTorch implementation!
python 10_pytorch_transformer.py
```

## 📖 What Each Lesson Covers

### Lesson 1: Neural Network Basics
**Real-World Analogy:** Restaurant Order System

Learn how a single neuron detects patterns, then combine neurons into layers:
- Dot products (pattern matching)
- Bias (baseline adjustment)
- ReLU activation (firing threshold)
- Softmax (probability distribution)

### Lesson 2: Embeddings
**Real-World Analogy:** Library Card Catalog

Learn how words become mathematical vectors:
- Token embeddings (word meaning)
- Position embeddings (word order)
- **NEW**: Structured embeddings showing semantic similarity

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

### Lesson 9: Mini GPT (NumPy)
**Real-World Implementation:** Complete working model!

Everything comes together in a working Mini GPT that:
- Tokenizes text (character-level)
- Processes through transformer blocks
- Generates text with multiple strategies

### Lesson 10: PyTorch Transformer ⭐ NEW!
**MCP-Enhanced Lesson** using official PyTorch documentation!

Bridge from educational to production code:
- `nn.MultiheadAttention` vs our NumPy implementation
- `nn.TransformerEncoderLayer` for complete blocks
- **Real backpropagation** with `torch.autograd`
- Training loop with actual gradient descent
- Text generation with trained model

## 🤖 MCP-Enhanced Course

This course was enhanced using Model Context Protocol (MCP) servers:

- **Context7 MCP**: Fetched official PyTorch documentation for Lesson 10
- **Sequential Thinking MCP**: Planned course structure and enhancements
- **Git MCP**: Version control and GitHub integration

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

### Architecture Comparison

| Component | GPT-2 Small | Our Mini GPT (NumPy) | Our Mini GPT (PyTorch) |
|-----------|-------------|---------------------|------------------------|
| Parameters | 124M | ~100K | ~1M |
| Embedding | 768 | 64 | 128 |
| Heads | 12 | 4 | 4 |
| Blocks | 12 | 2 | 4 |
| Backprop | ✓ | Demo only | ✓ Real! |
| GPU | ✓ | ✗ | ✓ Ready |

**Same architecture, different scales and capabilities!**

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

4. **Real Training with Lesson 10**
   ```python
   # Run the PyTorch lesson for real backprop
   python 10_pytorch_transformer.py
   
   # See loss decrease over training steps!
   # Step  Loss       Perplexity
   # 0      7.1834     1317.32
   # 5      7.0027     1099.55
   # 9      6.9402     1033.01
   ```

## 📚 Additional Resources

### Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer paper
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 details

### Blogs
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Visual explanation
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy's minimal GPT

### Next Steps
1. **Study Backpropagation** - How models actually learn (Lesson 10!)
2. **Explore BPE Tokenization** - How GPT-2/3 tokenize text
3. **Try nanoGPT** - Minimal PyTorch GPT implementation
4. **Hugging Face Course** - Free transformers course

## 🎉 What You've Accomplished

After completing this course:

✅ You understand how GPT works at a fundamental level
✅ You can explain self-attention, embeddings, and transformers
✅ You've built a working language model from scratch (NumPy)
✅ You understand generation strategies (greedy, sampling, top-k, top-p)
✅ You've seen real backpropagation training (PyTorch Lesson 10)
✅ You're ready to dive into more advanced deep learning!

## ⚠️ Important Notes

**NumPy Implementation (Lessons 1-9):**
- Educational implementation
- Forward pass only (no backpropagation)
- Character-level tokenizer
- Great for understanding concepts

**PyTorch Implementation (Lesson 10):**
- Production-ready code
- Real backpropagation with autograd
- GPU acceleration ready
- Bridge to real-world LLM development

For production use, see:
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Minimal PyTorch GPT
- [Hugging Face Transformers](https://huggingface.co/transformers/) - Production library

---

**Happy Learning! 🚀**

Start with `python 01_neural_network_basics.py` and work through each lesson!

## 📝 Repository

**GitHub**: https://github.com/brajrajnagar/learn-gpt-from-scratch

Latest updates:
- Added Lesson 10: PyTorch Transformer with real backprop
- Fixed embeddings with semantic structure (Lesson 2)
- MCP-enhanced documentation