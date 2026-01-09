# AI Campus Assistant 

An **offline AI Campus Assistant** that allows you to upload campus related PDFs (FAQs, rules, guides) and ask natural language questions about them. **No OpenAI or paid APIs required**.

---
##  Tech Stack

* **Python 3.10+**
* **Ollama** (local LLM runtime)
* **LangChain**
* **ChromaDB** (vector store)
* **PDF ingestion** via LangChain loaders

---

## Running the Project

### Step 1: Start Ollama Server

Open a terminal:

```bash
ollama serve
```

---

### Step 2: Run the Assistant

In another terminal (inside the project directory):

```bash
python assistant.py
```

---

##  How to Use

### Upload a PDF

Place your PDF in your home directory or project folder, then type:

```
upload ~/SHINE-FAQs.pdf
```

✔ First upload may take time (embedding creation)
✔ Data is saved in `chroma_db/`

---

### Ask Questions

After upload:

```
What is this document describing?
What are the campus rules?
Who can apply for scholarships?
```

---

## API Keys & Security

* No OpenAI key required
* No internet required after setup
* Safe to push to GitHub

---

## Use Cases

* Campus FAQ bots
* College helpdesk assistant
* Policy document Q&A
* Internal knowledge base

---

## Future Improvements

* Web UI (React / Flask)
* Source citations
* Streaming responses

---

## Summary

* 100% local & free
* Fast after first run
* Persistent memory
* Easy to extend
