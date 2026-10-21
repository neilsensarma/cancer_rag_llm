from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import json, os, chromadb
from tqdm import tqdm
hf_token = os.getenv("HF_TOKEN")

def get_chunky(splitter, document):
    return splitter.split_text(document)

def embed(model, chunk):
    embeddings = model.encode(chunk)
    return embeddings

def save_json(data):
    with open('../data/embedded_data_2.json', 'w') as f:
        json.dump(data, f)

def create_vector_db(embedded_data, chunk_ids, metadatas, chunked_documents):
    client = chromadb.PersistentClient(path="../data/chroma_db")
    collection=client.get_or_create_collection(name="cancer_bioinformatics", metadata={"hnsw:space": "cosine"})
    batch_size = 10000
    
    for start in range(0, len(embedded_data), batch_size):
        end = start+batch_size
        embedded_batch = embedded_data[start:end]
        chunk_batch = chunk_ids[start:end]
        metadata_batch = metadatas[start:end]
        chunked_doc_batch = chunked_documents[start:end]
        collection.add(embeddings = embedded_batch, ids = chunk_batch, metadatas = metadata_batch, documents = chunked_doc_batch)
    print(f'Stored Collection: {collection.count()}')
    
if __name__ == '__main__':
    with open('../data/raw_data/data_2.json', 'r') as f:
        data = json.load(f)
    
    metadata, data_for_embedding = {}, {}
    for i, (_, pmid_data) in enumerate(data.items()):
        pmid = pmid_data['pmid']
        metadata[pmid] = {'pmcid': pmid_data['pmcid'], 'authors': pmid_data['authors'], 'title':pmid_data['title'], 'journal': pmid_data['journal'], 'publish_date': pmid_data['pubdate']}
        data_for_embedding[pmid] = "\n\n".join([passage["text"] for passage in pmid_data['passages']])
    
    
    text_splitter = RecursiveCharacterTextSplitter(separators=["\n\n", "\n", " ", ""], chunk_size=1000, chunk_overlap=150, length_function=len, is_separator_regex=False)
    sentence_embed_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    
    chunked_data = []
    print("Chunking..")
    for _, (pmid, document) in enumerate(data_for_embedding.items()):
        doc_chunks = get_chunky(text_splitter, document)
        for i, chunk in enumerate(doc_chunks):
            chunked_data.append({f"{pmid}_{i}": chunk})
    print("Chunking Done..")
    
    save_data, embedded_chunks, chunked_ids, chunked_metadatas, chunked_documents = [], [], [], [], []
    print("Embedding..")
    for chunk in tqdm(chunked_data):
        chunk_id, current_chunk = next(iter(chunk.items()))
        pmid = chunk_id.split("_")[0]
        chunked_documents.append(current_chunk)
        embedded_chunk = embed(sentence_embed_model, current_chunk)
        chunked_ids.append(chunk_id)
        chunked_metadatas.append({"pmid": pmid, "chunk_id": int(chunk_id.split("_")[1]), "document": current_chunk, "author": ", ".join(metadata[pmid]['authors']), "title": metadata[pmid]['title'], "journal": metadata[pmid]['journal'], "publish_date": metadata[pmid]['publish_date']})
        embedded_chunks.append(embedded_chunk.tolist())
        save_data.append({"id": chunk_id, "embedding": embedded_chunk.tolist(), "document": current_chunk, "metadata": {"pmid": pmid, "chunk_id": int(chunk_id.split("_")[1]), "pmcid": metadata[pmid]['pmcid'], "author": metadata[pmid]['authors'], "title": metadata[pmid]['title'], "journal": metadata[pmid]['journal'], "publish_date": metadata[pmid]['publish_date']}})
    print("Embedding Done..")
    
    """ print(len(chunked_ids))
    print(len(embedded_chunks))
    print(len(chunked_metadatas))
    print(len(chunked_documents)) """
    
    #save_json(save_data)
    
    create_vector_db(embedded_data=embedded_chunks, chunk_ids=chunked_ids, metadatas=chunked_metadatas, chunked_documents=chunked_documents)
    print('Vector DB has been created')
    