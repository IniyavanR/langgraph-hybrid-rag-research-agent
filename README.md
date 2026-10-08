# 🔬 LangGraph Hybrid RAG Research Agent

An AI/ML research assistant I built using **LangGraph, ChromaDB, and LLMs** to explore how a RAG system can retrieve and reason over research papers.

Instead of building a basic:

```text
Question → Vector Search → LLM → Answer
```

pipeline, I wanted to understand what happens when retrieval isn't good enough.

So this project includes **query understanding, query expansion, MMR retrieval, bibliography filtering, Cross-Encoder reranking, relevance scoring, retrieval routing, web search, conversation memory, and retrieval evaluation**.

The main goal was not just to make something that answers questions, but to actually **measure how well the retrieval system works**.

---

## 🎯 What I Built

The agent works with a collection of AI/ML research papers and can answer questions based on them.

For example:

> "How does LoRA reduce the number of trainable parameters?"

The system doesn't immediately send this question to an LLM.

It first tries to understand the question, improve the query, retrieve relevant parts of the research papers, rerank them, and check whether the retrieved information is actually useful.

If the research-paper retrieval isn't strong enough, the system can use **Tavily web search** as an additional source of information.

It can also remember previous messages, so follow-up questions don't have to repeat all of their context.

---

# 🧠 How It Works

At a high level, the workflow looks like this:

```text
                         User Question
                              │
                              ▼
                       Query Maker
                              │
                              ▼
                        LLM Router
                         /       \
                        /         \
                  GENERIC       RESEARCH
                     │              │
                     ▼              ▼
              Generic Answer   Query Analysis
                                     │
                                     ▼
                              Query Resolution
                                     │
                                     ▼
                              Query Expansion
                                     │
                                     ▼
                              MMR Retrieval
                                     │
                                     ▼
                           Bibliography Filter
                                     │
                                     ▼
                             Cross-Encoder
                               Reranking
                                     │
                                     ▼
                            Hybrid Scoring
                                     │
                                     ▼
                           Retrieval Decision
                              /          \
                             /            \
                           RAG            WEB
                            │              │
                            └──────┬───────┘
                                   ▼
                            Answer Generation
                                   │
                                   ▼
                           Answer + Sources
```

The workflow is implemented as a **LangGraph state graph**, so the different stages can make decisions about what should happen next.

---

# 📚 Research Paper Knowledge Base

I created a private knowledge base from a collection of AI/ML research papers.

Some of the topics covered include:

* Transformers
* BERT
* GPT-3
* RAG
* InstructGPT / RLHF
* ResNet
* AlexNet
* GANs
* VAEs
* LoRA
* Diffusion models

The PDFs are loaded, chunked, embedded, and stored in **ChromaDB**.

---

# ✂️ Document Processing

One of the things I wanted to experiment with was **semantic chunking** instead of simply splitting every document into fixed-size pieces.

The basic pipeline is:

```text
Research Papers
      ↓
PDF Loading
      ↓
Semantic Chunking
      ↓
Large Chunk Splitting
      ↓
Small Chunk Filtering
      ↓
Embeddings
      ↓
ChromaDB
```

The semantic chunker uses a percentile-based breakpoint strategy.

After chunking, very small chunks are removed and oversized chunks are split further.

Each chunk also gets its own ID, which becomes useful later when evaluating retrieval.

---

# 🔎 Retrieval

The retrieval pipeline is probably the part of this project I spent the most time experimenting with.

It isn't just a similarity search.

### 1. Query Expansion

The original question is expanded with related terms before retrieval.

This helps when the user's wording doesn't exactly match the terminology used in a research paper.

```text
User Query
    ↓
Related terms
    ↓
Expanded Query
```

---

### 2. MMR Retrieval

The expanded query is sent to ChromaDB using **Maximal Marginal Relevance (MMR)**.

Current configuration:

```text
k = 20
fetch_k = 60
lambda_mult = 0.85
```

I used MMR because I didn't want the top results to simply be several almost-identical chunks.

---

### 3. Bibliography Filtering

Research papers contain a lot of references.

Those references can sometimes look relevant to a query even though they don't actually explain the concept we're looking for.

So I added a filtering step to remove bibliography-heavy chunks before reranking.

---

### 4. Cross-Encoder Reranking

After the initial retrieval, I use:

```text
cross-encoder/ms-marco-MiniLM-L-6-v2
```

to rerank the retrieved chunks.

The idea is:

```text
Vector Search
    ↓
Fast candidate retrieval
    ↓
Cross-Encoder
    ↓
More precise ranking
```

---

# ⚖️ Hybrid Relevance Score

I didn't want to rely entirely on either the vector similarity score or the Cross-Encoder score.

So the final relevance score combines both:

```text
Final Score =
    0.7 × Cross-Encoder Score
  + 0.3 × Similarity Score
```

The top candidates are then used to decide whether the retrieved context is strong enough to answer the question.

I also use two thresholds:

```text
Strong: ≥ 0.7

Weak:   ≤ 0.4
```

This gives the system a way to distinguish between useful retrieval and weak retrieval instead of assuming that every retrieved chunk is good.

---

# 🌐 Web Search

The local research papers obviously can't contain everything.

For that reason, I integrated **Tavily** as an external search source.

The idea is roughly:

```text
                 Query
                   │
                   ▼
             Local Retrieval
                   │
                   ▼
          Is the evidence useful?
              /           \
            Yes            No
             │              │
             ▼              ▼
            RAG          Web Search
```

The generated answer keeps the research-paper evidence and web evidence distinguishable so that the source of information isn't hidden.

---

# 💬 Conversation Memory

I also wanted the agent to behave more like an actual research assistant rather than a collection of independent queries.

For example:

```text
User:
What is LoRA?

Agent:
...

User:
Why is it parameter efficient?
```

The second question can use the previous conversation to understand what **"it"** refers to.

The graph uses **LangGraph state management and SQLite checkpointing** to maintain conversation state.

---

# 📌 Sources

When answering from the research-paper knowledge base, the system keeps track of the source and chunk ID.

For example:

```text
Source: LoRA
Chunk ID: semantic_chunk_123
```

This makes it possible to trace retrieved information back to the original document.

It also gives the retrieval benchmark a concrete way to identify whether the correct evidence was retrieved.

---

# 📊 Evaluation

I didn't want to evaluate the project only by asking:

> "Does the answer look good?"

Because an LLM can sometimes produce a correct-looking answer even when the retriever failed.

So I created a small manually verified retrieval benchmark using **gold chunks**.

The current benchmark contains **10 questions**.

### Retrieval Results

| Metric    |    Result |
| --------- | --------: |
| Recall@5  | **66.7%** |
| Recall@10 | **81.7%** |
| MRR       | **73.5%** |
| Hit@5     |   **90%** |
| Hit@10    |  **100%** |

The most useful result for me here is the Hit@10 score:

**100% of the benchmark questions had at least one relevant chunk within the top 10 retrieved chunks.**

The results also show that there is still room to improve the ranking because Recall@5 is lower than Recall@10.

---

# 🧪 RAGAS Evaluation

I also experimented with **RAGAS** to evaluate the generated answers.

The evaluation includes:

* Faithfulness
* Answer Relevancy
* Context Precision
* Context Recall
* Answer Correctness

Running LLM-based evaluation introduced some practical problems such as token limits and large evaluation prompts.

Rather than hiding those issues, I kept the evaluation work as part of the project because dealing with these constraints is itself useful experience when building LLM applications.

---

# 🛠️ Tech Stack

| Area            | Technology                               |
| --------------- | ---------------------------------------- |
| Language        | Python                                   |
| Agent framework | LangGraph                                |
| LLM             | Groq — `openai/gpt-oss-120b`             |
| Embeddings      | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector DB       | ChromaDB                                 |
| Retrieval       | MMR                                      |
| Reranking       | `cross-encoder/ms-marco-MiniLM-L-6-v2`   |
| Web Search      | Tavily                                   |
| Memory          | SQLite + LangGraph Checkpointing         |
| Evaluation      | Retrieval metrics + RAGAS                |

---

# 📁 Project Structure

```text
langgraph-hybrid-rag-research-agent/
│
├── LangGraph_project_code.ipynb
│
├── LangGraph_project_pdf_data/
│   ├── research_paper_1.pdf
│   ├── research_paper_2.pdf
│   └── ...
│
├── ChromaDB/
│
├── .gitignore
│
└── README.md
```

The main implementation is currently contained in:

```text
LangGraph_project_code.ipynb
```

---

# 🚀 Running the Project

### Clone the repository

```bash
git clone https://github.com/IniyavanR/langgraph-hybrid-rag-research-agent.git

cd langgraph-hybrid-rag-research-agent
```

### Create a virtual environment

```bash
python -m venv venv
```

On Windows:

```bash
venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Add API keys

Create a `.env` file:

```env
GROQ_API_KEY_2=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

Then open:

```text
LangGraph_project_code.ipynb
```

and run the notebook.

---

# 🧩 What I Learned From This Project

This project started as a way for me to learn **LangGraph and RAG**, but it ended up teaching me a lot more about the retrieval side of LLM applications.

Some of the main things I worked with were:

* Designing stateful LangGraph workflows
* Conditional graph routing
* RAG architecture
* Semantic chunking
* Embeddings
* ChromaDB
* MMR retrieval
* Cross-Encoder reranking
* Query expansion
* Retrieval thresholds
* Hybrid retrieval
* Conversation memory
* SQLite checkpointing
* Retrieval evaluation
* RAGAS
* Working around LLM inference/token limitations

One of the biggest takeaways was that **RAG isn't just about putting documents into a vector database**.

The quality of the final answer depends heavily on what happens *before* the LLM sees the context.

---

# 🔭 What's Next?

There are several things I'd like to improve in the next version:

* Expand the research-paper collection
* Build a larger retrieval benchmark
* Improve retrieval routing
* Compare different embedding models
* Compare chunking strategies
* Benchmark the effect of reranking
* Improve web-source validation
* Add stronger citation verification
* Add an evaluator/critic node
* Add a proper follow-up-question loop
* Build a UI and deploy the agent

---

# 👨‍💻 About

Built by **Iniyavan R** as an AI/ML portfolio project while learning and experimenting with **RAG, LangGraph, LLM applications, and retrieval systems**.

The goal of the project was simple:

> **Don't just make an LLM answer questions. Understand what happens between the question and the answer.**

---

⭐ If you found the project interesting, feel free to explore the notebook and the retrieval experiments.
