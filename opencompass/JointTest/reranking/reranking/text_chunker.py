from typing import List

from llama_index.core.data_structs import Node
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.response_synthesizers import CompactAndRefine, TreeSummarize
from llama_index.core.schema import NodeWithScore
from llama_index.core.async_utils import run_async_tasks
from llama_index.core import get_response_synthesizer


def synthesize(response_mode, query_str, text_chunks, scores):
    """ ResponseMode.REFINE, ResponseMode.COMPACT, ResponseMode.SIMPLE_SUMMARIZE, ResponseMode.TREE_SUMMARIZE
        ResponseMode.NO_TEXT, ResponseMode.GENERATION, ResponseMode.ACCUMULATE, ResponseMode.COMPACT_ACCUMULATE """
    nodes = [NodeWithScore(node=Node(text=text_chunks[i]), score=scores[i]) for i in range(len(text_chunks))]

    response = get_response_synthesizer(response_mode=response_mode).synthesize(query_str, nodes=nodes)
    print(response)
    if response_mode == ResponseMode.NO_TEXT:
        print(response.source_nodes)


# def repack(mode: str, query: str, docs: List[str], pack_size: int=3, top_k: int=1000) -> List[str]:
#     text_chunks_list = partition_docs(docs, pack_size, top_k)
def partition_docs(docs: List[str], pack_size: int, top_k: int):
    text_chunks_list = []

    if len(docs) <= pack_size:
        return [docs]

    for i in range(pack_size - 1, len(docs)):
        text_chunks = docs[:(pack_size - 1)]
        text_chunks.append(docs[i])
        text_chunks_list.append(text_chunks)
        if len(text_chunks_list) >= top_k:
            break

        if i > pack_size - 1 and pack_size > 1:
            text_chunks = docs[i - pack_size + 1: i + 1]
            text_chunks_list.append(text_chunks)
            if len(text_chunks_list) >= top_k:
                break

    return text_chunks_list


def repack(mode: str, query: str, docs: List[str], max_size: int=100, min_size: int=0, similarity_scores: List[float]=None, score_threshold: float=None, ordering='normal') -> str:
    # If similarity_scores & score_threshold are assigned, return docs with score >= score_threshold
    if similarity_scores is not None and score_threshold is not None:
        k = min(len(docs), max_size)
        for i in range(len(similarity_scores)):
            if similarity_scores[i] < score_threshold:
                k = min(i, k)
                break
        # Return at least min_size if assigned
        if min_size > 0:
            k = max(k, min_size)
    # If min_size is assigned, only return a minimum num of docs
    elif min_size > 0:
        k = min(len(docs), min_size)
    # Else, return at most max_size docs
    else:
        k = min(len(docs), max_size)

    repacked_docs = docs[:k]
    if ordering == 'normal':        # Normal reranked order, from the highest similarity to lowest
        pass
    elif ordering == 'reverse':     # Reverse the reranked order, from the lowest to highest
        repacked_docs.reverse()
    elif ordering == 'sides':       # Lost in the middle, [1 3 5 7 9 10 8 6 4 2]  [1 3 5 7 9 8 6 4 2]
        if k > 2:
            temp_docs = []
            if k % 2 == 0:
                for i in range(0, k - 1, 2):
                    temp_docs.append(repacked_docs[i])
                for i in range(k - 1, 0, -2):
                    temp_docs.append(repacked_docs[i])
            else:
                for i in range(0, k, 2):
                    temp_docs.append(repacked_docs[i])
                for i in range(k - 2, 0, -2):
                    temp_docs.append(repacked_docs[i])
            repacked_docs = temp_docs

    if mode == "compact":
        # Compact
        # cr = CompactAndRefine()
        # return cr._make_compact_text_chunks(query, repacked_docs)[0]
        
        words_as_str = [str(word) for word in repacked_docs]
        return "\n\n".join(words_as_str)

    if mode == "summarize":
        # TreeSummarize
        ts = TreeSummarize()
        summary_template = ts._summary_template.partial_format(query_str=query)
        if ts._use_async:
            if ts._output_cls is None:
                tasks = [
                    ts._llm.apredict(summary_template, context_str=text_chunk)
                    for text_chunk in repacked_docs
                ]
            else:
                tasks = [
                    ts._llm.astructured_predict(ts._output_cls, summary_template, context_str=text_chunk)
                    for text_chunk in repacked_docs
                ]

            summary_responses = run_async_tasks(tasks)

            if ts._output_cls is not None:
                summaries = [summary.json() for summary in summary_responses]
            else:
                summaries = summary_responses
        else:
            if ts._output_cls is None:
                summaries = [
                    ts._llm.predict(summary_template, context_str=text_chunk)
                    for text_chunk in repacked_docs
                ]
            else:
                summaries = [
                    ts._llm.structured_predict(ts._output_cls, summary_template, context_str=text_chunk)
                    for text_chunk in repacked_docs
                ]
                summaries = [summary.json() for summary in summaries]
        return ts._prompt_helper.repack(summary_template, text_chunks=summaries)[0]


if __name__ == '__main__':
    query = "who proposed the geocentric theory"
    docs = [
        "The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.opernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.",
        "Copernicus proposed a heliocentric model of the solar system â a model where everything orbited around the Sun. Today, with advancements in science and technology, the geocentric model seems preposterous.he geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.",
        "Nicolaus Copernicus (b. 1473âd. 1543) was the first modern author to propose a heliocentric theory of the universe. From the time that Ptolemy of Alexandria (c. 150 CE) constructed a mathematically competent version of geocentric astronomy to Copernicusâs mature heliocentric version (1543), experts knew that the Ptolemaic system diverged from the geocentric concentric-sphere conception of Aristotle.",
        "The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.ou might want to check out one article on the history of the geocentric model and one regarding the geocentric theory. Here are links to two other articles from Universe Today on what the center of the universe is and Galileo one of the advocates of the heliocentric model.",
        "For Earth-centered it was  Geocentric Theory proposed by greeks under the guidance of Ptolemy and Sun-centered was Heliocentric theory proposed by Nicolas Copernicus in 16th century A.D. In short, Your Answers are: 1st blank - Geo-Centric Theory. 2nd blank - Heliocentric Theory.",
        "The geocentric model, also known as the Ptolemaic system, is a theory that was developed by philosophers in Ancient Greece and was named after the philosopher Claudius Ptolemy who lived circa 90 to 168 A.D. It was developed to explain how the planets, the Sun, and even the stars orbit around the Earth.imple tools, such as the telescope â which helped convince Galileo that the Earth was not the center of the universe â can prove that ancient theory incorrect. You might want to check out one article on the history of the geocentric model and one regarding the geocentric theory.",
        ]
    scores = [-0.00836, -0.01693, -0.02351, -0.02447, -0.02536, -0.02599, -0.03345, -0.03409, -0.05057, -0.95129]

    results = repack("compact", query, docs, 100, 0, scores, -0.02)
    print(results)

    # synthesize(ResponseMode.REFINE, query_str, text_chunks, scores)
    # synthesize(ResponseMode.COMPACT, query_str, text_chunks, scores)
    # synthesize(ResponseMode.SIMPLE_SUMMARIZE, query_str, text_chunks, scores)
    # synthesize(ResponseMode.TREE_SUMMARIZE, query_str, text_chunks, scores)
    # synthesize(ResponseMode.NO_TEXT, query_str, text_chunks, scores)
    # synthesize(ResponseMode.GENERATION, query_str, text_chunks, scores)
    # synthesize(ResponseMode.ACCUMULATE, query_str, text_chunks, scores)
    # synthesize(ResponseMode.COMPACT_ACCUMULATE, query_str, text_chunks, scores)

