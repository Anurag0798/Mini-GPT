# Mini-GPT

A minimal implementation of a GPT-style (Generative Pre-trained Transformer) language model using TensorFlow and Keras. This project is designed for educational purposes and small-scale experimentation-train your own mini generative language model from scratch or on your own data, with all core logic transparent and customizable.

## Features

- **Customizable Transformer Decoder Model:**  
  Build a GPT-style model with configurable number of layers, embedding size, attention heads, and feed-forward dimensions.
- **Train from Scratch:**  
  Tokenize text, build input sequences, and train your own GPT on provided or custom text data (`data.txt`).
- **Text Generation Utility:**  
  Generate text from a prompt/token sequence with temperature and top-k sampling controls.
- **Easy-to-read Code:**  
  Clean, well-commented logic for core transformer blocks, positional encoding, and the entire training loop.
- **Jupyter Notebook Based:**  
  Run, edit, and experiment cell-by-cell.  
- **No External Dependence on Hugging Face:**  
  All core GPT-like mechanics implemented in TensorFlow/Keras.

## Repository Structure

```
.
├── mini_gpt.ipynb   # Main notebook with the model, training, and generation pipeline
├── data.txt         # Training corpus (add your own text here)
└── README.md        # This file
```


## Requirements

- Python 3.7+
- tensorflow (2.x)
- numpy

You can install the necessary packages via pip:

```bash
pip install tensorflow numpy
```

## Getting Started

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Anurag0798/Mini-GPT.git

   cd Mini-GPT
   ```

2. **(Optional) Set up and activate a virtual environment:**
   ```bash
   python -m venv .venv

   # On macOS/Linux
   source .venv/bin/activate

   # On Windows
   .venv\Scripts\Activate.ps1
   ```

3. **Install dependencies:**
   ```bash
   pip install tensorflow numpy
   ```

4. **Prepare your data:**
   - Edit/add your text data in `data.txt`.
   - The notebook reads from this file during preprocessing.

5. **Run the notebook:**
   ```bash
   jupyter notebook
   ```
   - Open `mini_gpt.ipynb` and step through the code for training and text generation.


## How it Works

### 1. Data Preparation
- The notebook tokenizes text in `data.txt`, prepares training sequences, and creates input/target data for next-token prediction.

### 2. Model Construction
- Implements a positional encoding layer.
- Stacks multiple transformer decoder blocks (Multi-Head Self-Attention + FeedForward).
- Ends with a Dense projection to the vocabulary size (softmax output).

### 3. Training
- Customizable model size: number of layers, heads, embedding dimension, feed-forward size, etc.
- Compiles with Adam optimizer and categorical cross-entropy loss.
- Trains for the specified number of epochs and batch size.

### 4. Text Generation
- From a seed prompt, repeatedly predicts the next token using top-k and temperature sampling.
- Decodes the output sequence back to text.
- All core logic for token sampling and sequence management is included and easy to modify.

## Customization

- *Change the model size:* Edit parameters like `num_layers`, `embed_dim`, `num_heads`, and `ff_dim` in the notebook.
- *Train on your own data:* Replace or append to `data.txt` with your own corpus.
- *Experiment with sampling:* Tune `temperature`, `top_k`, or max sequence length for text generation diversity and control.



## Limitations & Recommendations

- This is a minimal implementation, best suited for small datasets or educational use.
- For large datasets/language modeling tasks, use GPU acceleration (e.g., Colab).
- Not intended for production-scale training or serving.
- For real-world NLP problems with large scale, refer to Hugging Face Transformers.

## License

Please refer to the LICENSE file for more details.

## Contributing

Contributions are welcome! You can:
- Extend the model (add more features, configurable options)
- Report issues or bugs
- Suggest improvements

Feel free to fork the repo and open a pull request.