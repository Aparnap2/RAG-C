# Installation

## Basic Installation

```bash
pip install RAG-C
```

## Provider-Specific Installation

### Vector Stores

=== "Qdrant"
    ```bash
    pip install "RAG-C[qdrant]"
    ```

=== "AstraDB"
    ```bash
    pip install "RAG-C[astradb]"
    ```

### Text Indexes

=== "OpenSearch"
    ```bash
    pip install "RAG-C[opensearch]"
    ```

=== "Elasticsearch"
    ```bash
    pip install "RAG-C[elasticsearch]"
    ```

### Complete Installation

```bash
pip install "RAG-C[all]"
```

## Development Installation

```bash
git clone https://github.com/yourusername/RAG-C.git
cd RAG-C
make install-dev
```

## Requirements

- Python 3.9+
- Vector store (Qdrant, AstraDB)
- Text index (OpenSearch, Elasticsearch) - optional
- LLM API key (Google GenAI, OpenRouter)

## Verification

```python
import RAG-C
print(RAG-C.__version__)
```