from pyserini.search.lucene import LuceneSearcher
import json


def run_bm25():
    searcher = LuceneSearcher.from_prebuilt_index('msmarco-v1-passage')
    hits = searcher.search('what is a lobster roll?', k=1000)

    for i in range(0, 1000):
        print(f'{i + 1:2} {hits[i].docid:7} {hits[i].score:.5f}')


def filter_qrels_w_queries():
    input_file = '../data/msmarco/passage/dev-6980/base-queries.jsonl'
    qrel_file = '../data/msmarco/passage/dev-6980/qrels.dev.small.tsv'
    output_file = '../data/msmarco/passage/dev-6980/qrels_with_queries.dev.small.tsv'

    with open(input_file, 'r') as infile, open(qrel_file, 'r') as qfile, open(output_file, 'w') as outfile:
        qid_set = set()
        for line in infile:
            data = json.loads(line)
            query_id = data['query_id']
            qid_set.add(query_id)

        added_set = set()
        for line in qfile:
            qid = line.split('\t')[0]
            # if qid in qid_set and qid not in added_set:
            if qid in qid_set:
                added_set.add(qid)
                outfile.write(line)

    print("Conversion complete!")


def filter_queries_w_qrels():
    input_file = '../data/msmarco/passage/dev-6980/base-queries.jsonl'
    qrel_file = '../data/msmarco/passage/dev-6980/qrels.dev.small.tsv'
    output_file = '../data/msmarco/passage/dev-6980/queries.dev.small.tsv'

    with open(input_file, 'r') as infile, open(qrel_file, 'r') as qfile, open(output_file, 'w') as outfile:
        query_w_qrels_set = set()
        for line in qfile:
            query_w_qrels_set.add(line.split('\t')[0])

        for line in infile:
            data = json.loads(line)
            query_id = data['query_id']
            query = data['query']

            if query_id in query_w_qrels_set:
                outfile.write(f"{query_id}\t{query}\n")

    print("Conversion complete!")   # Seems like qrels file is complete, after filter still 6980 entries


if __name__ == '__main__':
    # filter_queries_w_qrels()
    filter_qrels_w_queries()