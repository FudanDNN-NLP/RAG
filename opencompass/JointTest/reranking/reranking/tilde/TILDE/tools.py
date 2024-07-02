from tqdm import tqdm
import re


def load_run(run_path, run_type='trec'):
    run = {}
    with open(run_path, 'r') as f:
        for line in tqdm(f, desc="loading run...."):
            if run_type == 'msmarco':
                qid, docid, score = line.strip().split("\t")
            elif run_type == 'trec':
                qid, _, docid, rank, score, _ = line.strip().split(" ")
            qid = qid
            docid = docid
            if qid not in run.keys():
                run[qid] = []
            run[qid].append(docid)
    return run


def load_collection(collection_path):
    collection = {}
    with open(collection_path, 'r') as f:
        for line in tqdm(f, desc="loading collection...."):
            docid, text = line.strip().split("\t")
            collection[docid] = text
    return collection


def load_queries(query_path):
    query = {}
    with open(query_path, 'r') as f:
        for line in tqdm(f, desc="loading query...."):
            qid, text = line.strip().split("\t")
            query[qid] = text
    return query


def get_batch_text(start, end, docids, collection):
    batch_text = []
    for docid in docids[start: end]:
        batch_text.append(collection[docid])
    return batch_text


def get_stop_ids(tok):
    # hard code for now, from nltk.corpus import stopwords, stopwords.words('english')
    stop_words = set(['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', "you're", "you've",
                      "you'll", "you'd", 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his',
                      'himself', 'she', "she's", 'her', 'hers', 'herself', 'it', "it's", 'its', 'itself',
                      'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this',
                      'that', "that'll", 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been',
                      'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                      'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for',
                      'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after',
                      'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under',
                      'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all',
                      'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not',
                      'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                      "don't", 'should', "should've", 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren',
                      "aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't",
                      'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't", 'ma', 'mightn', "mightn't", 'mustn',
                      "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't",
                      'weren', "weren't", 'won', "won't", 'wouldn', "wouldn't"])
    # keep some common words in ms marco questions
    stop_words.difference_update(["where", "how", "what", "when", "which", "why", "who"])

    vocab = tok.get_vocab()
    tokens = vocab.keys()

    stop_ids = []

    for stop_word in stop_words:
        ids = tok(stop_word, add_special_tokens=False)["input_ids"]
        if len(ids) == 1:
            stop_ids.append(ids[0])

    for token in tokens:
        token_id = vocab[token]
        if token_id in stop_ids:
            continue
        if token == '##s':  # remove 's' suffix
            stop_ids.append(token_id)
        if token[0] == '#' and len(token) > 1:  # skip most of subtokens
            continue
        if not re.match("^[A-Za-z0-9_-]*$", token):  # remove numbers, symbols, etc..
            stop_ids.append(token_id)

    return set(stop_ids)
