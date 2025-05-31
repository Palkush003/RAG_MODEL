### 🤖 Neurific AI RAG Chatbot

A best-in-class Retrieval-Augmented Generation (RAG) chatbot for answering questions about Long Short-Term Memory (LSTM) networks, featuring beautiful mathematical formula rendering, authoritative sources, and a modern Streamlit web UI.

---

## 🚀 Features

- **Completely Free & Local:** No cloud APIs, all models and data run on your machine.
- **State-of-the-Art RAG Pipeline:** Combines local LLMs (Ollama), Sentence Transformers, and ChromaDB.
- **Beautiful Math Rendering:** LSTM equations displayed in LaTeX/Markdown for clarity.
- **Authoritative Sourcing:** Answers cite Chris Olah’s blog and CMU LSTM notes.
- **Smart Retrieval & Reranking:** Uses MMR and quality-weighted chunk selection.
- **Performance Dashboard:** See response time, cache hits, and answer quality.
- **Modern UI:** Streamlit chat interface with message history and quick actions.

---

## 📦 Installation

### Prerequisites

- Python 3.11 or 3.12
- [Ollama](https://ollama.com/) (for local LLM inference)
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) (optional, for image OCR)

### Setup

```bash
git clone https://github.com/Palkush003/RAG_MODEL.git
cd Internship
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
```

### Ollama Setup

1. Download and install Ollama from [ollama.com](https://ollama.com/)
2. Start Ollama:
   ```
   ollama serve
   ```
3. Pull the Llama 3.2 model:
   ```
   ollama pull llama3.2
   ```

---

## 💻 Usage

### Streamlit Web App

```bash
streamlit run streamlit_app.py
```

- Ask any LSTM question in the chat.
- For LSTM equations, type: `mathematical formulation of LSTM` or use the quick action button.

### Command Line

```bash
python rag-chatbot-main.py
```

---

## 📝 Example Questions

- What is the vanishing gradient problem and how do LSTMs solve it?
- How does the forget gate work in an LSTM?
- mathematical formulation of LSTM
- What are the differences between LSTM and standard RNN?
- How do LSTMs handle long-term dependencies?

---

## 🗂️ Project Structure

```
neurific-ai-rag-chatbot/
├── rag-chatbot-main.py       # Main backend chatbot
├── streamlit_app.py          # Streamlit web frontend
├── requirements.txt
├── cache/                    # Cached web and PDF content
├── enhanced_chroma_db/       # Vector DB for Chroma
└── README.md                 # This file
```

---

## 📚 Data Sources

- [Understanding LSTM Networks by Chris Olah](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [CMU Deep Learning LSTM Notes](https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf)

---

## 🧠 Technical Stack

- **LangChain** for RAG pipeline orchestration
- **Ollama** for local LLM inference (Llama 3.2)
- **Sentence Transformers** for semantic embeddings
- **ChromaDB** for vector search
- **Streamlit** for the web UI

---

## 🛠️ Advanced Features

- **10/10 Answer Quality:** Prompt engineering, reranking, and post-processing for perfect answers.
- **LaTeX Math Rendering:** LSTM formulas displayed in clean, readable format.
- **Performance Metrics:** Track response time, cache hit rate, and more.
- **Quick Action Buttons:** For common LSTM queries in the web UI.

---

## 🐞 Troubleshooting

- **Ollama not running:**  
  Run `ollama serve` in a terminal.
- **Model not found:**  
  Run `ollama pull llama3.2`
- **Import errors:**  
  Run `pip install -r requirements.txt`
- **Windows build errors:**  
  Install [Microsoft Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)

---

## 🤝 Contributing

Contributions are welcome! Please open issues or pull requests for bug fixes, features, or documentation improvements.


---

## 🙏 Acknowledgements

- Chris Olah for his foundational LSTM blog
- CMU for open-sourcing their deep learning notes
- [LangChain](https://github.com/langchain-ai/langchain), [Ollama](https://ollama.com/), [Chroma](https://www.trychroma.com/), [Streamlit](https://streamlit.io/)

---

## ⭐️ Star this repo if you find it useful!

---

> _Built with ❤️ by Palkush Dave

