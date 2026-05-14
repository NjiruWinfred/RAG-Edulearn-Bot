# 🎓 AI Education Chatbot (RAG-Based)

An AI-powered education chatbot designed to reduce hallucinations by grounding responses in curriculum-specific content using Retrieval-Augmented Generation (RAG).

This project was built as my capstone after completing 6 months of training in Python, SQL, Excel, and Power BI. It focuses on applying real-world AI concepts such as vector search, embeddings, and contextual retrieval.

---

## 🚀 Project Overview

The chatbot is designed for an education app where students can ask questions and receive answers based on structured curriculum data rather than generic internet responses.

To achieve this, I implemented a **RAG pipeline**, where the system retrieves relevant curriculum content before generating a response.

A key enhancement in this system is its **hybrid offline capability**, allowing parts of the application to function even without internet connectivity.

---

## 🧠 Key Features

- 📚 Curriculum-aware question answering
  
- 🔍 Retrieval-Augmented Generation (RAG)
  
- 💬 Conversational AI chatbot interface
  
- 🗂️ Chat history stored in MongoDB
  
- 🧠 Vector storage using MongoDB
  
- ⚡ Fast retrieval of contextual information
  
- 📴 **Hybrid offline mode for limited or no internet environments**

---

## 📴 Offline Feature (Hybrid Mode)

This system includes a lightweight offline design to improve accessibility in low-connectivity environments:

- Cached student and curriculum data stored locally
- Pre-fetched embeddings stored in MongoDB for fast retrieval
- Local fallback responses when API/LLM is unavailable
- Loop-based logic to serve stored knowledge without requiring internet access

### How it works:
1. User query is received
2. System first checks local cache (offline store)
3. If match is found → returns cached response instantly
4. If not found → system switches to online RAG pipeline
5. Results are stored for future offline use

This reduces dependency on constant internet access and improves reliability in real-world educational settings.

---

## 🏗️ Tech Stack

- Python
- MongoDB (chat history + vector storage)
- AI/LLM API (Google API)
- RAG architecture (Embeddings + Retrieval + Generation)
- Backend API (FastAPI)

---

## ⚙️ How It Works

1. User sends a question
2. Question is converted into embeddings
3. Relevant curriculum content is retrieved from MongoDB vector storage
4. Retrieved context is combined with the user query
5. LLM generates a grounded response using that context
6. Response is returned and stored in chat history

---

## 📦 Project Structure

```bash
.
├── app.py / main.py
├── requirements.txt
├── models/
├── utils/
├── data/
├── embeddings/
└── README.md

**🚧 Deployment (Issue Faced)
**
The project is deployed on Render, but currently faces a startup issue:

❌ “No open ports detected”

Initial investigation suggests this may be caused by:

Long RAG initialization during startup

Embedding/vector loading blocking server startup

MongoDB connection delays

Server not binding correctly to 0.0.0.0:$PORT

**🧪 What I’m Learning From This Project**

Practical implementation of RAG systems

Vector databases and embeddings

Real-world deployment challenges

Debugging production-level Python applications

Building hybrid online/offline AI systems

**📌 Future Improvements**

Move embedding generation to background process

Optimize startup time for deployment

Improve retrieval accuracy

Add evaluation metrics for hallucination reduction

Deploy on more scalable infrastructure

Strengthen offline-first architecture

**🙏 Acknowledgements**

This is my first end-to-end AI project combining backend development, machine learning concepts, and deployment. Any feedback or guidance is highly appreciated.

**📬 Contact**

If you have suggestions or improvements, feel free to connect or open an issue.

