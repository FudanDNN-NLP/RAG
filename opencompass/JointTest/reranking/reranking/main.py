from typing import List, Tuple
import time

from .dlm.pygaggle.pygaggle.rerank.base import Query, Text
from .dlm.pygaggle.pygaggle.rerank.transformer import MonoT5, MonoBERT
from pyserini.search import LuceneSearcher
from .dlm.pygaggle.pygaggle.rerank.base import hits_to_texts

from .tilde.run_rerank import tilde_rerank

from .dlm.rankllama import rankllama_rerank

from .dlm.transformer import MonoT5Zh
from .text_chunker import repack


def rerank(model, expansion_model, tokenizer, bert_tokenizer, mode, query: str, docs: List[str], top_k: int = None) -> Tuple[List[str], List[float]]:
    if top_k is None:
        top_k = len(docs)

    reranked_docs, similarity_scores = run_rerank(model, expansion_model, tokenizer, bert_tokenizer, mode, query, docs, top_k)
    return reranked_docs, similarity_scores


def run_rerank(model, expansion_model, tokenizer, bert_tokenizer, mode, query, docs, top_k=None):

    if top_k is None or top_k > len(docs):
        top_k = len(docs)

    if mode == "MonoT5" or mode == "MonoBERT":  # Reranker models: MonoT5, MonoBERT, DuoT5
        # if mode == "MonoBERT":
        #     model = MonoBERT()
        # else:
        #     model = MonoT5()
        model = model
        query = Query(query)
        if docs is None:
            # Option 1: fetch some passages to rerank from MS MARCO with Pyserini
            # searcher = LuceneSearcher.from_prebuilt_index('msmarco-passage')
            # searcher = LuceneSearcher.from_prebuilt_index('msmarco-doc')  # download index, 10G
            searcher = LuceneSearcher("indexes/index-msmarco-passage-20191117-0ed488")  # load local index
            hits = searcher.search(query.text)
            texts = hits_to_texts(hits)
        else:
            # Option 2: <passages> what Pyserini would have retrieved, hard-coded
            # Note, pyserini scores don't matter since T5 will ignore them.
            # texts = [Text(doc[1], {'docid': doc[0]}, 0) for doc in docs]
            texts = [Text(doc) for doc in docs]

        # The passages prior to reranking:
        # print("=============== Before Reranking ===============")
        # for i in range(top_k):
        #     print(f'{i + 1:2} {texts[i].score:.5f} {texts[i].text}')

        # Rerank:
        reranked = model.rerank(query, texts)

        # Reranked results:
        print("=============== After Reranking ===============")
        reranked_docs, scores = [], []
        for i in range(top_k):
            reranked_docs.append(reranked[i].text)
            scores.append(float(f"{reranked[i].score:f}"))
            print(f"{i + 1:2} {reranked[i].score:.5f} {reranked[i].text}")

        return reranked_docs, scores

    elif mode == "RankLLaMA":
        query = Query(query)
        reranked_docs, scores = rankllama_rerank(model, tokenizer, query, docs)

        # Reranked results:
        print("=============== After Reranking ===============")
        for i in range(top_k):
            print(f"{i + 1:2} {scores[i]:.5f} {reranked_docs[i]}")
        return reranked_docs[:top_k], scores[:top_k]

    elif mode == "TILDE":
        reranked_docs, scores = tilde_rerank(model, expansion_model, tokenizer, bert_tokenizer, query, docs)

        print("=============== After Reranking ===============")
        for i in range(top_k):
            print(f"{i + 1:2} {scores[i]:.5f} {reranked_docs[i]}")
        return reranked_docs[:top_k], scores[:top_k]

    elif mode == "MonoT5Zh":
        model = MonoT5Zh()

        texts = [Text(doc) for doc in docs]

        # Rerank:
        reranked = model.rerank(query, texts)

        # Reranked results:
        print("=============== After Reranking ===============")
        reranked_docs, scores = [], []
        for i in range(top_k):
            reranked_docs.append(reranked[i].text)
            scores.append(float(f"{reranked[i].score:f}"))
            print(f"{i + 1:2} {reranked[i].score:.5f} {reranked[i].text}")

        return reranked_docs, scores

    start_time = time.time()

    # 1. Base MonoT5 reranking
    mode = "MonoT5"

    query1 = "who proposed the geocentric theory"
    passages1 = [
        "For Earth-centered it was  Geocentric Theory proposed by greeks under the guidance of Ptolemy and Sun-centered was Heliocentric theory proposed by Nicolas Copernicus in 16th century A.D. In short, Your Answers are: 1st blank - Geo-Centric Theory. 2nd blank - Heliocentric Theory.",
        "Copernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.he geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.",
        "The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.opernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.",
        "Copernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â which helped convince Galileo that the Earth was not the center of the universe â can prove that ancient theory incorrect.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.",
        "Copernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.Simple tools, such as the telescope â which helped convince Galileo that the Earth was not the center of the universe â can prove that ancient theory incorrect.opernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.",
        "The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.imple tools, such as the telescope â which helped convince Galileo that the Earth was not the center of the universe â can prove that ancient theory incorrect. You might want to check out one article on the history of the geocentric model and one regarding the geocentric theory.",
        "Nicolaus Copernicus (b. 1473âd. 1543) was the first modern author to propose a heliocentric theory of the universe. From the time that Ptolemy of Alexandria (c. 150 CE) constructed a mathematically competent version of geocentric astronomy to Copernicusâs mature heliocentric version (1543), experts knew that the Ptolemaic system diverged from the geocentric concentric-sphere conception of Aristotle.",
        "A Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth. Geocentric theory is an astronomical theory which describes the universe as a Geocentric system, i.e., a system which puts the Earth in the center of the universe, and describes other objects from the point of view of the Earth.",
        "The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.",
        "After 1,400 years, Copernicus was the first to propose a theory which differed from Ptolemy's geocentric system, according to which the earth is at rest in the center with the rest of the planets revolving around it.",
    ]
    reranked_docs, scores = rerank(mode, query1, passages1, 100)

    # 2. Different query
    # query1 = 'tell me about Alexandria'
    # run_rerank(mode, query1, passages1)
    #
    # query2 = 'what\'s the size of fudan'
    # passages2 = [
    #     'Establishment: Fudan University is one of the oldest and most prestigious universities in China, founded in 1905 in Shanghai.',
    #     'Rankings: It consistently ranks among the top universities in China and globally. It\'s often ranked within the top 5 universities in China and in the top 100 globally according to various international rankings.',
    #     'Multidisciplinary Approach: Fudan University is known for its comprehensive academic programs spanning various disciplines including humanities, social sciences, natural sciences, engineering, and medicine.',
    #     'Internationalization: Fudan has a strong focus on internationalization, with partnerships and collaborations with over 200 universities worldwide. It\'s home to a diverse student body and faculty, including a significant number of international students and scholars.',
    #     'Research Output: The university is renowned for its research output and contributions to various fields. It houses numerous research institutes and laboratories conducting cutting-edge research in areas such as biotechnology, nanotechnology, economics, and more.',
    #     'Campus: Fudan University\'s main campus is located in the Yangpu District of Shanghai, spanning over 2.8 square kilometers. The campus features modern facilities, libraries, sports centers, and green spaces.',
    #     'Notable Alumni: Fudan has produced many notable alumni, including political leaders, business tycoons, academics, and cultural figures. Some of its alumni include Jiang Zemin (former General Secretary of the Communist Party of China), Guo Moruo (renowned poet and historian), and Wang Huning (member of the Politburo Standing Committee of the Communist Party of China).',
    #     'Libraries: The university boasts extensive library collections, with Fudan University Library being one of the largest academic libraries in China. It houses millions of volumes of books, journals, and digital resources catering to various academic disciplines.',
    #     'Social Impact: Fudan University plays a significant role in shaping social and economic developments in China and beyond. Its research and academic contributions influence policies, industries, and society at large.',
    #     'Global Initiatives: Fudan actively engages in global initiatives such as joint research projects, student exchange programs, and international conferences. These initiatives enhance cross-cultural understanding and academic collaboration on a global scale.'
    # ]
    # run_rerank(mode, query2, passages2)
    #
    # query2 = 'what courses can you find at fudan'
    # run_rerank(mode, query2, passages2)

    # 3. Different modes
    mode = "MonoBERT"
    reranked_docs, scores = rerank(mode, query1, passages1, 100)
    #
    # mode = "RankLLaMA"
    # reranked_docs, scores = rerank(mode, query1, passages1, 100)

    # 4. With compact repacking, ranking no reverse & reverse
    repacked_large_doc = repack("compact", query1, reranked_docs, 100, 0, scores, -0.025)
    print("\n--- Repacked Results ---\n")
    print(repacked_large_doc)
    repacked_large_doc = repack("compact", query1, reranked_docs, 100, 0, scores, -0.025, True)
    print("\n--- Reversed Repacked Results ---\n")
    print(repacked_large_doc)

    end_time = time.time()
    print(f"Run time: {end_time - start_time}s")
