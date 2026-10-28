from rag_new import Embedder, VectorDB, ReRanker, DataHelper
import os
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
hf_token = os.getenv("HF_TOKEN")

def evaluation(ground_truth, retrieved_chunks, k):
    act_answer = list(ground_truth.keys())
    retrieved_answers = [i['chunk_id'] for i in retrieved_chunks][:k]
    true_positives = len(set(act_answer) & set(retrieved_answers))
    precision = true_positives/len(retrieved_answers)
    recall = true_positives/len(act_answer)
    reciprocal_rank = 0.0
    for rank, chunk_id in enumerate(retrieved_answers, start=1):
        if chunk_id in act_answer:
            reciprocal_rank = 1/rank
            break
    return precision, recall, reciprocal_rank
    
def main():
    embedding_model = Embedder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
    retriever = VectorDB()
    helper = DataHelper()
    reranker = ReRanker(CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device='mps'))
    
    embedded_data = helper.read_json('../data/embeddings/embedded_data_2.json')
    chunk_lookup = helper.create_lookup(embedded_data)
    test_queries = helper.read_json('../data/eval_data/rag_test_queries.json')
    test_answers = helper.read_json('../data/eval_data/rag_qrels_known_positives.json')
    recall_25 = []
    metrics = {'cosine_similarity': {'precision': [], 'recall': [], 'mrr': []}, 'reranked': {'precision': [], 'recall': [], 'mrr': []}}
    for q in tqdm(test_queries):
        #get the question and embed the question
        query = q['query']
        embedded_q = embedding_model.encode_sentence(query)
        
        #cosine similarity based retrieval
        similarity_scores = retriever.get_similarity_scores(embedded_q, embedded_data)
        top_k_chunks = retriever.get_top_k_chunks(similarity_scores, 25)
        _, rec, _ = evaluation(test_answers[q['query_id']], top_k_chunks, 25)
        recall_25.append(rec)
        
        #perform evaluation of the cosine retrieved chunks
        pre_cosine, rec_cosine, mrr_cosine = evaluation(test_answers[q['query_id']], top_k_chunks, 5)
        metrics['cosine_similarity']['precision'].append(pre_cosine)
        metrics['cosine_similarity']['recall'].append(rec_cosine)
        metrics['cosine_similarity']['mrr'].append(mrr_cosine)
        
        #perform reranking of the cosine retrieved documents
        reranked_documents = [rank[1] for rank in reranker.rerank(query, top_k_chunks,chunk_lookup, 5)]
        
        #perform evaluation of the reranked retrieved documents
        pre_rerank, rec_rerank, mrr_rerank = evaluation(test_answers[q['query_id']], reranked_documents, 5)
        metrics['reranked']['precision'].append(pre_rerank)
        metrics['reranked']['recall'].append(rec_rerank)
        metrics['reranked']['mrr'].append(mrr_rerank)

    print(f"Average Cosine Similarity Precision: {sum(metrics['cosine_similarity']['precision'])/len(metrics['cosine_similarity']['precision'])}")
    print(f"Average Cosine Similarity Recall: {sum(metrics['cosine_similarity']['recall'])/len(metrics['cosine_similarity']['recall'])}")
    print(f"Average Cosine Similarity MRR: {sum(metrics['cosine_similarity']['mrr'])/len(metrics['cosine_similarity']['mrr'])}")
    print("-------")
    print(f"Average Reranked Precision: {sum(metrics['reranked']['precision'])/len(metrics['reranked']['precision'])}")
    print(f"Average Reranked Recall: {sum(metrics['reranked']['recall'])/len(metrics['reranked']['recall'])}")
    print(f"Average Reranked MRR: {sum(metrics['reranked']['mrr'])/len(metrics['reranked']['mrr'])}")
    print(f"Recall: {sum(recall_25)/len(recall_25)}")
    
if __name__ == '__main__':
    main()