# Impulse Intelligent Model (IIM)

The **Impulse Intelligent Model (IIM)** is a research-driven AI project designed to demonstrate how modern AI systems can combine:

- **Machine Learning**
- **Retrieval-Augmented Reasoning**
- **Structured Tool Usage**

to perform complex cognitive tasks such as:

- Code generation  
- Technical reasoning  
- Knowledge-based question answering  
- Task planning  
- Codebase understanding  

The project was developed for the **Impulse AI Model Track Hackathon** with the objective of building a **fully functional AI model integrated into the Impulse Machine Learning architecture**.

---

# Overview

The **Impulse Intelligent Model (IIM)** is a **multi-capability AI system** powered by a **transformer-based reasoning model** trained through the **Impulse ML training pipeline**.

The system integrates three core capabilities:

### Natural Language Understanding
Enables the model to interpret human instructions and contextual queries.

### Code Intelligence
Allows the system to generate, analyze, and reason about source code.

### Knowledge Retrieval
Enhances responses using embedding-based retrieval from external knowledge sources.

These capabilities are orchestrated through a **modular AI architecture** that allows the model to reason about complex tasks while dynamically interacting with external knowledge systems.

---

# Key Features

## Reasoning Engine
Supports **multi-step reasoning** for complex analytical queries.

## Code Generation
Produces **syntactically correct and functional code snippets** across programming languages.

## Knowledge Retrieval
Uses **embedding-based vector search** to augment model knowledge.

## Modular ML Pipeline
Training and inference run entirely within the **Impulse Machine Learning framework**.

## API-Driven Inference
A **FastAPI service** exposes the model for real-time interaction and integration with external applications.

---

# Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-org/Impulse-intelligent_model
cd Impulse-intelligent_model

## 2. Backend Setup

Navigate to the backend directory:

```bash
cd backend
```

Install required dependencies:

```bash
pip install -r requirements.txt
```

---

## 3. Start Vector Database (Optional)

The retrieval system uses a **vector database** for embedding-based search.

Start the Qdrant container:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

---

## Training the Model

Run the training pipeline:

```bash
python training/train_model.py
```

The training pipeline performs the following operations:

- Dataset loading  
- Tokenization and preprocessing  
- Transformer model optimization  
- Model checkpoint generation  

---

## Running Inference

Start the inference API server:

```bash
uvicorn api.server:app --reload
```

This launches the **FastAPI service** used to interact with the model.

---

## Example Query

### Endpoint

```
POST /generate
```

### Request

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that implements quicksort."
  }'
```

### Response

```json
{
  "response": "def quicksort(arr): ..."
}
```

---

## Demo Instructions

### Step 1 — Train the Model

```bash
python training/train_model.py
```

### Step 2 — Start the Inference Server

```bash
uvicorn api.server:app
```

### Step 3 — Send an API Request

```bash
curl -X POST "http://localhost:8000/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Write a Python function that implements quicksort."
  }'
```

### Step 4 — Observe Model Output

The system will generate responses based on **reasoning, retrieval, and model inference**.

---

## Evaluation

The system is evaluated across three capability benchmarks:

| Capability | Metric |
|------------|--------|
| Reasoning  | Logical QA accuracy |
| Coding     | Code generation correctness |
| Knowledge  | Retrieval accuracy |

Evaluation scripts are located in:

```
evaluation/
```

---

## Future Work

Planned improvements include:

- Multi-agent reasoning architecture  
- Reinforcement learning fine-tuning  
- Codebase graph memory  
- IDE integrations for developer workflows  

---

## License

This project is released under the **Apache2.0 Liscense**.

---

## Acknowledgements

This project builds on open research and engineering practices in:

- Transformer architectures  
- Retrieval-Augmented Generation (RAG)  
- Modern machine learning pipelines