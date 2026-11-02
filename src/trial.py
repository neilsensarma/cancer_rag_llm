from Bio import Entrez

Entrez.email = "your_email@example.com"

query = "cryptochrome 4 AND magnetoreception"

with Entrez.esearch(
    db="pubmed",
    term=query,
    retmax=5,
    sort="relevance"
) as handle:
    results = Entrez.read(handle)

print("Total results:", results["Count"])
print("PMIDs:", results["IdList"])
print("Query translation:", results.get("QueryTranslation"))