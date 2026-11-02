from tqdm import tqdm
import numpy as np
from llama_cpp import Llama
import json, chromadb, os
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sentence_transformers.cross_encoder import CrossEncoder
from Bio import Entrez, Medline
import warnings

warnings.filterwarnings("ignore")
hf_token = os.getenv("HF_TOKEN")
ncbi_token = os.getenv("NCBI_API")
Entrez.email = "nilavrasensarma@gmail.com"
Entrez.api_key = ncbi_token

class DataHelper:
    '''This class helps with data reading and manipulation'''
    @staticmethod
    def read_json(data):
        with open(data, 'r') as f:
            data = json.load(f)
        return data
    
    @staticmethod
    def create_lookup(embedded_data):
        return {data["id"]: data for data in embedded_data}

class Embedder:
    '''This class helps with embedding the query sentence'''
    def __init__(self, encoder):
        self.encoder = encoder
    
    def encode_sentence(self, sentence):
        return self.encoder.encode(sentence)

class ChromaDBRetrieval:
    def __init__(self, path, collection_name):
        self.client = chromadb.PersistentClient(path = path, settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_collection(name = collection_name)
    
    def chroma_retrieval(self, query, k):
        results = self.collection.query(query_texts = [query], n_results = k)
        ids, scores, metadatas = results['ids'][0], results['distances'][0], results['metadatas'][0]
        chunk_scores = list(zip(ids, scores))
        for i, chunk in enumerate(metadatas):
                chunk_id = str(chunk['pmid'])+'_'+str(chunk['chunk_id'])
                chunk['ids'] = chunk_id
                chunk['scores'] = chunk_scores[i][1]
        return metadatas

class LocalRetriever:
    '''This class computes the similarity between the query and the local knowledge'''
    def cosine_similarity(self, a, b):
        dp = sum(x*y for x, y in zip(a, b))
        norm_a = sum(x*x for x in a) ** 0.5
        norm_b = sum(y*y for y in b) ** 0.5
        return dp / (norm_a * norm_b)
    
    def get_top_k_chunks(self, scores, k):
        sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
        top_k_chunks = sorted_scores[:k]
        return top_k_chunks

    def get_similarity_scores(self, embedded_input, embedded_chunks):
        similarity_scores = []
        for chunk in embedded_chunks:
            current_embedding, chunk_id = chunk['embedding'], chunk['id']
            similarity_scores.append({'chunk_id': chunk_id, 'score': self.cosine_similarity(embedded_input, current_embedding)})
        return similarity_scores
    
class ReRanker:
    '''This class reranks the original similar documents'''
    def __init__(self, model):
        self.model = model 
        
    def rerank(self, query, top_k_chunks, top_n = 5):
        pairs = [[query, chunk['document']] for chunk in top_k_chunks]
        reranked_scores = self.model.predict(pairs, show_progress_bar = False)
        return sorted(list(zip(reranked_scores, top_k_chunks)), key=lambda x: x[0], reverse=True)[:top_n]
        
class Context: 
    '''This class generates the context for the Llama model to generate the answer'''
    def generate(self, reranked_documents):
        input_for_rag = []
        for chunk in reranked_documents:
                #chunk_id = chunk['chunk_id']
                input_for_rag.append({'date': chunk['publish_date'].split(" ")[0] ,'pmid': chunk['pmid'], 'title': chunk['title'], 'authors': chunk['author'], 'document': chunk['document']})    
        context = "\n\n".join(["\n".join([f"PMID: {chunk['pmid']}", f"Title: {chunk['title']}", f"Authors: {chunk['authors']}", f"Year of Publication: {chunk['date']}", f"Text: {chunk['document']}"]) for chunk in input_for_rag])
        return context
    
class AI:
    '''This class checks if the local context is enough to answer the query and also reformulates the query'''
    def __init__(self, model):
        self.model = model
        
    def reformulate_query(self, query):
        response = self.model.create_chat_completion(
            messages = [
                {
                    'role': 'system',
                    'content': (
                    "You generate short PubMed keyword searches. "
                    "Extract only the 2 or 3 most important scientific concepts "
                    "from the user's question. "
                    "Join them using AND. "
                    "Use plain words only. "
                    "Do not use field tags, quotation marks, parentheses, or OR. "
                    "Do not try to represent every detail of the question. "
                    "Example question: How do migratory birds detect magnetic fields "
                    "using cryptochrome proteins? "
                    "Example output: cryptochrome AND magnetoreception. "
                    "Return only the search expression."
                )
                },
                {
                    'role': 'user',
                    'content': query
                }
            ],
            temperature = 0.0,
            max_tokens=400
        )
        return response['choices'][0]['message']['content']
    
    def web_scrapping_tool(self, query):
        #set up the entrez api caller to retrieve the relevant papers
        #returns the most relevant document 
        with Entrez.esearch(db="pubmed", term=query, retmax=5, sort="relevance") as handle:
            results = Entrez.read(handle)
            pmids = ",".join(results['IdList'])
        
        if not pmids:
            return []

        with Entrez.efetch(db="pubmed", id=pmids, rettype="medline", retmode="text") as handle:
            records = list(Medline.parse(handle))
        
        papers = []
        for record in records:
            papers.append({"Date": record.get("DP", ), "PMID": record.get("PMID"), "Author": record.get("AU", []), "Title": record.get("TI"), "Text": record.get("AB", "")})
        
        context = "\n\n".join([f"PMID: {paper['PMID']}, Author: {paper['Author']}, Title: {paper['Title']}, Text: {paper['Text']}, Date: {paper['Date']}" for paper in papers])
        return context
    
    def check_local_knowledge(self, query, context):
        #check here if the local knowledge is enough to answer the query
        #else call the web scraper tool
        #return the new context
        responses = self.model.create_chat_completion(
            messages = [
                {
                'role': 'system',
                'content': (
                    'You are a fact checking assistant between the provided context and the query. '
                    'Evaluate the sufficiency of the provided context without relying on your own pretrained knowledge. '
                    'If the context has enough scientific evidence to answer the query '
                    'then just answer YES spelled exactly like that. '
                    'If the context doesnt have enough scientific evidence then answer NO spelled exactly like that.' 
                ) 
                },
                {
                    'role': 'user',
                    'content': f'Context: {context}, query {query}. Check if the context is enough to answer the query'
                }
            ],
            temperature = 0.0, max_tokens=10
        )
        
        decision = responses['choices'][0]['message']['content'].strip().upper()
        
        if  'yes' in decision.strip().lower():
            print("Local knowledge is enough to answer this question.")
            return True
        elif 'no' in decision.strip().lower():
            print("Local knowledge is not enough to answer this")
            return False
        else:
            raise ValueError(f"Unexpected error found: {decision}")
        
    
class Generator:
    def __init__(self, model):
        self.model = model
    
    def generate_response(self, context, query):
        response = self.model.create_chat_completion(
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
                    'Do not include a concluding paragraph unless it adds new information. '
                    'List the references in a numbered manner.'
                )
            },
                {
                    'role': 'user',
                    'content':f'Context: {context} Question: {query}. Answer the question using the context provided to you'
                }
            ], 
            temperature = 0.2, max_tokens = 1000
        )
        
        return response
            
def main():
    #Initialize all models
    #embedding_model = Embedder(SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2'))
    #helper = DataHelper()
    #local_retriever = LocalRetriever()
    chromadb_retriever = ChromaDBRetrieval("../data/chroma_db", "cancer_bioinformatics")
    reranker = ReRanker(CrossEncoder("cross-encoder/ms-marco-MiniLM-L6-v2", device='mps'))
    llama = Llama(model_path = "../models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf", n_ctx=8192, n_gpu_layers=-1, verbose=False)
    ai = AI(llama)
    context_generator = Context()
    generator_model = Generator(llama)
    
    #define local answer as an empty string
    local_context, final_context = "", ""
    
    #get the query and embed it
    #query = "How can machine learning models classify malignant and non-malignant cells using gene expression signatures from single-cell RNA sequencing data"
    #query = "How do migratory birds use cryptochrome 4 proteins and quantum radical-pair mechanisms to detect Earth's magnetic field and navigate during migration?"
    query = "How does the gut microbiome influence the development of Parkinson's disease through the gut-brain axis, particularly through microbial metabolites, neuroinflammation, and alpha-synuclein aggregation?"
    #embedded_query = embedding_model.encode_sentence(query).tolist()
    
    """ #load in the embeddings
    embedded_data = helper.read_json('../data/embeddings/embedded_data_2.json')
    
    #get the top-k-chunks using similarity scores
    similarity_scores = local_retriever.get_similarity_scores(embedded_query, embedded_data)
    top_k_chunks = local_retriever.get_top_k_chunks(similarity_scores, k=25) """
    
    #use chromadb for cosine based similarity retrieval
    top_k_chunks = chromadb_retriever.chroma_retrieval(query, 25)
    #print(top_k_chunks[:5])
    
    """ #helpful for quick lookup of the metadata of the local context
    lookup = helper.create_lookup(embedded_data) """
    reranked_documents = [rank[1] for rank in reranker.rerank(query, top_k_chunks, 5)]
    #print(reranked_documents)
    
    #generate context using existing local knowledge
    local_context = context_generator.generate(reranked_documents)
    #print(local_context)
    
    #check if local context is enough
    if ai.check_local_knowledge(query, local_context):
        final_context = local_context 
    else:
        query_new = ai.reformulate_query(query)
        reformed_context = ai.web_scrapping_tool(query_new)
        
        final_context = local_context + "\n\n" + reformed_context
    
    if final_context:
        response = generator_model.generate_response(final_context, query)
        print(response['choices'][0]['message']['content'])
        print(response['usage'])
        print(response['choices'][0]['finish_reason'])
    else:
        print("I dont know, I am dumb")
    
if __name__ == "__main__":
    main()