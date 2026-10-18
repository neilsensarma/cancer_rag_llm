import requests
from tqdm import tqdm
import json

def retrieve_pmids():
    pmid_query = """
        (
            cancer[Title/Abstract]
            OR tumor[Title/Abstract]
            OR tumour[Title/Abstract]
            OR oncology[Title/Abstract]
        )
        AND
        (
            bioinformatics[Title/Abstract]
            OR "computational biology"[Title/Abstract]
            OR genomics[Title/Abstract]
            OR transcriptomics[Title/Abstract]
            OR "RNA sequencing"[Title/Abstract]
            OR "single cell"[Title/Abstract]
            OR "machine learning"[Title/Abstract]
        )
        AND
        pubmed pmc open access[filter]
        """
    pmid_url = ("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi")
    pmid_params = {
        "db": "pubmed",
        "term": pmid_query,
        "retmode": "json",
        "retmax": 1000,
        "sort": "relevance"
    }
    response = requests.get(pmid_url, pmid_params)
    response.raise_for_status()
    results = response.json()
    return results['esearchresult']['idlist']

def retrieve_documents(id):
    bmic_url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/{id}/unicode"
    
    response = requests.get(bmic_url)
    if response.status_code != 200:
        return None
    try:
        return response.json()
    except ValueError:
        return None
    
def retrieve_metadata(pmids):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    metadata = {}
    for start in tqdm(range(0, len(pmids), 100)):
        batch = pmids[start:start+100]
        response = requests.get(url, params={"db": "pubmed", "id": ",".join(batch), "retmode": "json"}, timeout=30)
        response.raise_for_status()
        result = response.json()['result']
        for pmid in batch:
            article = result.get(pmid)
            if article is None:
                continue
            article_ids = article.get("articleids", [])
            pmcid = next(
                (
                    identifier["value"]
                    for identifier in article_ids
                    if identifier.get("idtype") == "pmc"
                ),
                None
            )
            metadata[pmid] = {'pmid': pmid, 'pmcid': pmcid, 'title': article.get('title', ""), 'authors': [author["name"]
                    for author in article.get("authors", [])
                    if author.get("name")], "journal": article.get("fulljournalname", ""),
                              "pubdate": article.get("pubdate", "")}   
    return metadata

def extract_passages(raw_doc):
    allowed_sections = {"ABSTRACT", "INTRO", "METHODS", "RESULTS", "DISCUSS", "CONCL", "CASE", "APPENDIX"}

    extracted_passages = []

    for collection in raw_doc:
        for document in collection.get("documents", []):
            for passage in document.get("passages", []):

                infons = passage.get("infons", {})
                section = str(
                    infons.get("section_type", "")
                ).strip().upper()

                if section not in allowed_sections:
                    continue

                text = passage.get("text", "")

                if not text or not text.strip():
                    continue

                extracted_passages.append({
                    "section": section,
                    "text": text
                })

    return extracted_passages

if __name__ == '__main__':
    ids = retrieve_pmids()
    print("PMID's have been retrieved!")
    metadata_for_ids = retrieve_metadata(ids)
    print("Metadata have been retrieved!")
    raw_docs = [retrieve_documents(id) for id in tqdm(ids)] 
    doc_to_id_map = zip(ids, raw_docs)
    raw_docs_filt = [(ids, doc) for ids, doc in doc_to_id_map if doc is not None]
    act_ids = [item[0] for item in raw_docs_filt]
    act_docs = [item[1] for item in raw_docs_filt]
    structured_docs = {}

    for pmid, raw_doc in tqdm(raw_docs_filt):

        if pmid not in metadata_for_ids:
            continue

        passages = extract_passages(raw_doc)

        if not passages:
            continue

        structured_docs[pmid] = {
            **metadata_for_ids[pmid],
            "passages": passages
        }
    
    print("Total PMIDs:", len(ids))
    print("Metadata retrieved:", len(metadata_for_ids))
    print("BioC documents retrieved:", len(raw_docs_filt))
    print("Structured documents:", len(structured_docs))
    
    with open('../data/data_2.json', 'w+') as f:
        json.dump(structured_docs, f, indent=4)
    
    