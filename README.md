# Marco-RAG RAG-based Document Interaction System for local enviroments

This repository contains a Python application that utilizes a Retriever-Augmented Generation (RAG) model for document interaction. It allows users to chat with a set of documents, retrieving information and generating responses based on the content of these documents. This system is particularly useful for navigating large sets of documents, such as manuals, reports, or any collection of text documents, by asking questions in natural language.

## Features

- **Document Processing**: Automatically processes and indexes a given set of documents (.pdf, .docx) to make them searchable.
- **Natural Language Interaction**: Engage with the document collection through a chat interface by asking questions or making requests in natural language.
- **GPU/MPS Acceleration**: Utilizes GPU resources for efficient model inference, with support for Apple Silicon (M1/M2) MPS acceleration.
- **Memory Optimization**: Implements strategies for efficient memory usage, allowing for operation on systems with limited GPU memory.

## System Requirements

- Python 3.7+
- PyTorch 1.9.0+ (with MPS support for Apple Silicon)
- Transformers library by Hugging Face
- FAISS for efficient similarity search
- Additional Python libraries: `numpy`, `PyPDF2`, `python-docx`, `sentence-transformers`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-github/rag-document-interaction.git
   cd rag-document-interaction
   ```

2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the application, you need a folder containing your documents (.pdf, .docx). The system will process these documents, creating embeddings that allow the RAG model to retrieve relevant information in response to user queries.

1. Start the application by pointing it to your document folder:
   ```bash
   python main.py /path/to/your/documents
   ```

2. Once the application is running, you can interact with it by typing questions or queries related to the content of your documents. Type `quit` to exit the application.

## How It Works

1. **Document Processing**: The system first converts the text content of all provided documents into embeddings using a sentence transformer model.

2. **FAISS Indexing**: Embeddings are indexed using FAISS for efficient similarity search during the retrieval process.

3. **Query Handling**: When a query is received, the system converts it into an embedding and uses FAISS to find the most relevant document embeddings.

4. **Response Generation**: The RAG model uses the retrieved document content as context to generate a natural language response to the query.

## Customization

- **Model Selection**: You can change the underlying RAG model by modifying the `model_name` parameter in the `RAGModel` class.
- **Memory Management**: For systems with limited GPU memory, consider adjusting batch sizes or offloading computations to the CPU as described in the optimization section of the code.

## Contributing

Contributions to the project are welcome! Please refer to the `CONTRIBUTING.md` file for guidelines on how to contribute.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.
