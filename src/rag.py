from tqdm import tqdm
import numpy as np
from llama_cpp import Llama
import json, chromadb, os
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
hf_token = os.getenv("HF_TOKEN")

def read_json(data):
    with open(data, 'r') as f:
        data = json.load(f)
    return data

def cosine_similarity(a, b):
    dp = sum(x*y for x, y in zip(a, b))
    norm_a = sum(x*x for x in a) ** 0.5
    norm_b = sum(y*y for y in b) ** 0.5
    return dp / (norm_a * norm_b)

def get_similarity_scores(embedded_input, embedded_chunks):
    similarity_scores = []
    for chunk in tqdm(embedded_chunks):
        current_embedding, chunk_id = chunk['embedding'], chunk['id']
        similarity_scores.append({'chunk_id': chunk_id, 'score': cosine_similarity(embedded_input, current_embedding)})
    return similarity_scores

def get_top_k_chunks(scores, k):
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    top_k_chunks = sorted_scores[:k]
    return top_k_chunks

def reranker(query, retrieval_scores, lookup_dict, k=25, top_n = 5):
    model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device='mps')
    top_k_chunks = get_top_k_chunks(retrieval_scores, k)
    pairs = [[query, lookup_dict[chunk['chunk_id']]['document']] for chunk in top_k_chunks]
    reranked_scores = model.predict(pairs, show_progress_bar = False)
    return sorted(list(zip(reranked_scores, top_k_chunks)), key=lambda x: x[0], reverse=True)[:top_n]
    
def generate_response(context, query):
    model = Llama(model_path = "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", n_ctx=4096, n_gpu_layers=-1, verbose=False)
    response = model.create_chat_completion(
        messages = [
                    {
            'role': 'system',
            'content': (
                'You are a scientific research assistant specializing in '
                'cancer biology, bioinformatics, genomics, transcriptomics, '
                'machine learning, and computational biology. '
                'Use the context provided to you as scientific evidence to write the answer. '
                'Do not create fake citations, authors, year of publications or makeup answers if not provided in the context. '
                'Provide the correct citations when writing your answers. '
                'Answer briefly, as if summarizing and providing citations with each summary. '
                'Avoid repeating the same scientific findings across paragraphs. '
                'Do not include a concluding paragraph unless it adds new information. '
                'Write the names of the papers along with the author names and year of publication provided in the context in the form of a numbered list at the end.'
            )
        },
            {
                'role': 'user',
                'content':f'Context: {context} Question: {query}. Answer the question using the context provided to you'
            }
        ], 
        temperature = 0.2, max_tokens = 600
    )
    
    return response

def main():
    #input = input("Hi! What would you like to know today?\n")
    query = str("How can machine learning models classify malignant and non-malignant cells using gene expression signatures from single-cell RNA sequencing data")
    embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embedded_query = embedding_model.encode(query)
    embedded_data = read_json('../data/embeddings/embedded_data_2.json')
    
    #print(embedded_data[0])
    similarity_scores = get_similarity_scores(embedded_query, embedded_data)
    input_for_rag = []
    chunk_lookup = {data["id"]: data for data in embedded_data}
    reranked_documents = [rank[1] for rank in reranker(query, similarity_scores, chunk_lookup, 25, 5)]
        
    for chunk in reranked_documents:
        chunk_id = chunk['chunk_id']
        input_for_rag.append({'date': chunk_lookup[chunk_id]['metadata']['publish_date'].split(" ")[0] ,'pmid': chunk_lookup[chunk_id]['metadata']['pmid'], 'pmcid': chunk_lookup[chunk_id]['metadata']['pmcid'], 'title': chunk_lookup[chunk_id]['metadata']['title'], 'authors': ', '.join(chunk_lookup[chunk_id]['metadata']['author']), 'document': chunk_lookup[chunk_id]['document']})


    context = "\n\n".join(["\n".join([f"PMID: {chunk['pmid']}", f"Title: {chunk['title']}", f"Authors: {chunk['authors']}", f"Year of Publication: {chunk['date']}", f"Document: {chunk['document']}"]) for chunk in input_for_rag])
    response = generate_response(context, query)
    print(response['choices'][0]['message']['content'])
    print(response['usage'])
    print(response['choices'][0]['finish_reason'])
    
if __name__ == "__main__":
    main()