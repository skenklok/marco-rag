import argparse
import logging
from document_processor import process_documents, create_faiss_index, get_embeddings
from rag_model import RAGModel
import numpy as np
import torch


# Initialize logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def retrieve_documents(query, index, documents, model, k=5):
    with torch.no_grad():
        query_embeddings = model.model.get_input_embeddings()(
            model.tokenizer(query, return_tensors="pt")["input_ids"].detach().clone()
        ).cpu().numpy()
    query_embedding_aggregated = np.mean(query_embeddings, axis=1).squeeze(0)
    _, indices = index.search(query_embedding_aggregated.reshape(1, -1), k)
    return [documents[i] for i in indices[0]]

def chat_with_documents(index, texts, rag_model):
    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        relevant_docs = retrieve_documents(user_input, index, texts, rag_model)  # Ensure this returns a list
        # print("relevant_docs: ", relevant_docs)
        context = []  # Initialize context as an empty list

        # Append relevant documents to context, if any
        for doc in relevant_docs:
            context.append(doc)

        # Generate a response using the rag_model
        response = rag_model.generate_response(user_input, context)  # Now, context is guaranteed to be a list
        print("Bot:", response)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Chat with your documents.')
    parser.add_argument('folder_path', type=str, help='Path to the folder containing documents to chat with.')
    args = parser.parse_args()

    rag_model = RAGModel()
    texts, embeddings = process_documents(args.folder_path, rag_model)
    index = create_faiss_index(embeddings)
    chat_with_documents(index, texts, rag_model)

