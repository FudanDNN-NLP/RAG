python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input ./data/data/ \
  --index ./indexes/bm25_index/ \
  --generator DefaultLuceneDocumentGenerator \
  --threads 7 \
  --storePositions --storeDocvectors --storeRaw