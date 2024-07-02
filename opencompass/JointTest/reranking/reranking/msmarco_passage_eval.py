"""
This module computes evaluation metrics for MSMARCO dataset on the ranking task.
Command line:
python msmarco_eval_ranking.py <path_to_reference_file> <path_to_candidate_file>

Creation Date : 06/12/2018
Last Modified : 1/21/2019
Authors : Daniel Campos <dacamp@microsoft.com>, Rutger van Haasteren <ruvanh@microsoft.com>
"""
import os
import sys
from collections import Counter

MaxMRRRank = 1000


def load_reference(path_to_reference):
    """Load Reference reference relevant passages
    Args:path_to_reference (str): path to a file to load.
    Returns:qids_pids_gold (dict): dictionary mapping from query_id (int) to relevant passages (list of ints).
    """
    qids_pids_gold = {}
    with open(path_to_reference,'r') as file:
        for line in file:
            try:
                line = line.strip().split('\t')
                qid = int(line[0])
                if qid in qids_pids_gold:
                    pass
                else:
                    qids_pids_gold[qid] = []
                qids_pids_gold[qid].append(int(line[2]))
            except:
                raise IOError('\"%s\" is not valid format' % line)
    return qids_pids_gold


def load_candidate(path_to_candidate):
    """Load candidate data from a file.
    Args:path_to_candidate (str): path to file to load.
    Returns:qids_to_candidate_pids_ranking (dict): dictionary mapping from query_id (int) to a list of 1000 passage ids(int) ranked by relevance and importance
    """
    qids_to_candidate_pids_ranking = {}
    with open(path_to_candidate,'r') as file:
        for line in file:
            try:
                line = line.strip().split('\t')
                qid, pid, rank = int(line[0]), int(line[1]), int(line[2])
                if qid in qids_to_candidate_pids_ranking:
                    pass
                else:
                    # By default, all PIDs in the list of 1000 are 0. Only override those that are given
                    qids_to_candidate_pids_ranking[qid] = [0] * 1000
                qids_to_candidate_pids_ranking[qid][rank - 1] = pid
            except:
                raise IOError('\"%s\" is not valid format' % line)
    return qids_to_candidate_pids_ranking


def compute_metrics(qids_pids_gold, qids_to_candidate_pids_ranking):
    """Compute MRR metric
    Args:    
    p_qids_pids_gold (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_candidate_pids_ranking (dict): dictionary of query-passage candidates
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    all_scores = {}
    MRR_1, MRR_3, MRR_10, MRR_100, MRR, R_10 = 0, 0, 0, 0, 0, 0
    ranking = []
    for qid in qids_to_candidate_pids_ranking:
        if qid in qids_pids_gold:
            ranking.append(0)
            pid_gold = qids_pids_gold[qid]
            candidate_pids_ranking_list = qids_to_candidate_pids_ranking[qid]
            for i in range(MaxMRRRank):
                if candidate_pids_ranking_list[i] in pid_gold:
                    if i < 1:
                        MRR_1 += 1 / (i + 1)
                    if i < 3:
                        MRR_3 += 1 / (i + 1)
                    if i < 10:
                        MRR_10 += 1 / (i + 1)
                        R_10 += 1
                    if i < 100:
                        MRR_100 += 1 / (i + 1)
                    MRR += 1 / (i + 1)
                    ranking.pop()
                    ranking.append(i + 1)
                    break
    if len(ranking) == 0:
        raise IOError("No matching QIDs found. Are you sure you are scoring the evaluation set?")

    MRR_1 /= len(qids_pids_gold)
    MRR_3 /= len(qids_pids_gold)
    MRR_10 /= len(qids_pids_gold)
    MRR_100 /= len(qids_pids_gold)
    MRR /= len(qids_pids_gold)
    R_10 /= len(qids_pids_gold)

    all_scores['MRR@1 (P)'] = MRR_1
    all_scores['MRR@3'] = MRR_3
    all_scores['MRR@10'] = MRR_10
    all_scores['MRR@100'] = MRR_100
    all_scores['MRR'] = MRR
    all_scores['R@10'] = R_10
    all_scores['QueriesRanked'] = len(set(qids_to_candidate_pids_ranking))
    all_scores['Passages/Query'] = len(next(iter(qids_to_candidate_pids_ranking.values())))
    return all_scores


def quality_checks_qids(qids_pids_gold, qids_to_candidate_pids_ranking):
    """Perform quality checks on the dictionaries

    Args:
    p_qids_pids_gold (dict): dictionary of query-passage mapping
        Dict as read in with load_reference or load_reference_from_stream
    p_qids_to_candidate_pids_ranking (dict): dictionary of query-passage candidates
    Returns:
        bool,str: Boolean whether allowed, message to be shown in case of a problem
    """
    message = ''
    allowed = True

    # Create sets of the QIDs for the submitted and reference queries
    candidate_set = set(qids_to_candidate_pids_ranking.keys())
    ref_set = set(qids_pids_gold.keys())

    # Check that we do not have multiple passages per query
    for qid in qids_to_candidate_pids_ranking:
        # Remove all zeros from the candidates
        duplicate_pids = set([item for item, count in Counter(qids_to_candidate_pids_ranking[qid]).items() if count > 1])

        if len(duplicate_pids-set([0])) > 0:
            message = "Cannot rank a passage multiple times for a single query. QID={qid}, PID={pid}".format(
                    qid=qid, pid=list(duplicate_pids)[0])
            allowed = False

    return allowed, message


def compute_metrics_from_files(path_to_reference, path_to_candidate, perform_checks=True):
    """Compute MRR metric
    Args:    
    p_path_to_reference_file (str): path to reference file.
        Reference file should contain lines in the following format:
            QUERYID\tPASSAGEID
            Where PASSAGEID is a relevant passage for a query. Note QUERYID can repeat on different lines with different PASSAGEIDs
    p_path_to_candidate_file (str): path to candidate file.
        Candidate file sould contain lines in the following format:
            QUERYID\tPASSAGEID1\tRank
            If a user wishes to use the TREC format please run the script with a -t flag at the end. If this flag is used the expected format is 
            QUERYID\tITER\tDOCNO\tRANK\tSIM\tRUNID 
            Where the values are separated by tabs and ranked in order of relevance 
    Returns:
        dict: dictionary of metrics {'MRR': <MRR Score>}
    """
    
    qids_pids_gold = load_reference(path_to_reference)
    qids_to_candidate_pids_ranking = load_candidate(path_to_candidate)
    if perform_checks:
        allowed, message = quality_checks_qids(qids_pids_gold, qids_to_candidate_pids_ranking)
        if message != '': print(message)

    return compute_metrics(qids_pids_gold, qids_to_candidate_pids_ranking)


def main(path_to_reference=None, path_to_candidate=None):
    """ Command line:
    python msmarco_passage_eval.py <path_to_reference_file> <path_to_candidate_file>
    """
    # Args
    if len(sys.argv) == 1 and (path_to_reference is None or path_to_candidate is None):
        print("Usage: msmarco_passage_eval.py <path_to_reference_file> <path_to_candidate_file>")
        return
    if path_to_reference is None:
        if len(sys.argv) >= 2:
            path_to_reference = sys.argv[1]
    if path_to_candidate is None:
        if len(sys.argv) >= 3:
            path_to_candidate = sys.argv[2]

    # Evaluation
    metrics = compute_metrics_from_files(path_to_reference, path_to_candidate)

    # Output
    print('#####################')
    for metric in metrics:
        val = metrics[metric]
        if isinstance(val, float):
            val = '{:.5f}'.format(val)
        print('{}:\t{}'.format(metric, val))
    print('#####################')

    
if __name__ == '__main__':
    print("================  Random Ordering  ================")
    main("data/msmarco_ans_small/qrels.dev.small.tsv", "data/msmarco_ans_small/run.dev.small_shuffled.tsv")
    print("================  Before Reranking (BM25) ================")
    main("data/msmarco_ans_small/qrels.dev.small.tsv", "data/msmarco_ans_small/run.dev.small.tsv")
    print("================  After Reranking (MonoBERT)  ================")
    main("data/msmarco_ans_small/qrels.dev.small.tsv", "dlm/runs/run.monobert.ans_small.dev.tsv")
    print("================  After Reranking (MonoT5)  ================")
    main("data/msmarco_ans_small/qrels.dev.small.tsv", "dlm/runs/run.monot5.ans_small.dev.tsv")
    print("================  After Reranking (MonoT5 large)  ================")
    main("data/msmarco_ans_small/qrels.dev.small.tsv", "dlm/runs/run.monot5-large.ans_small.dev.tsv")
    print("================  After Reranking (RankLLaMA)  ================")
    main("data/msmarco_ans_small/qrels.dev.small.tsv", "dlm/runs/run.rankllama.ans_small.dev.tsv")
    print("================  After Reranking (TILDEv2)  ================")
    main("data/msmarco_ans_small/qrels.dev.small.tsv", "tilde/runs/TILDEv2.txt")

    print("================  Random Ordering  ================")
    main("data/msmarco/passage/dev-6980/qrels.dev.small.tsv", "data/msmarco/passage/dev-6980/run.dev.small_shuffled.tsv")
    print("================  Before Reranking Full Dev (BM25) ================")
    main("data/msmarco/passage/dev-6980/qrels_with_queries.dev.small.tsv", "data/msmarco/passage/dev-6980/run.dev.small.tsv")
    print("================  After Reranking Full Dev (MonoBERT)  ================")
    main("data/msmarco/passage/dev-6980/qrels_with_queries.dev.small.tsv", "dlm/runs/run.monobert.ans_full.dev.tsv")
    print("================  After Reranking Full Dev (MonoT5)  ================")
    main("data/msmarco/passage/dev-6980/qrels_with_queries.dev.small.tsv", "dlm/runs/run.monot5.ans_full.dev.tsv")
    print("================  After Reranking (TILDEv2)  ================")
    main("data/msmarco/passage/dev-6980/qrels_with_queries.dev.small.tsv", "tilde/runs/TILDEv2_full.txt")
    # print("================  Before Reranking (Zh)  ================")
    # main("data/msmarco/passage/cn/qrels.dev.small-sorted-100.tsv", "data/msmarco/passage/cn/run.dev.small-100.tsv")
    # print("================  After Reranking (T5-Zh 7k)  ================")
    # main("data/msmarco/passage/cn/qrels.dev.small-sorted-100.tsv", "dlm/runs/run.monot5.zh-7000.dev.tsv")
    # print("================  After Reranking (T5-Zh 17k)  ================")
    # main("data/msmarco/passage/cn/qrels.dev.small-sorted-100.tsv", "dlm/runs/run.monot5.zh-17000.dev.tsv")
    # print("================  After Reranking (T5-Zh 45k)  ================")
    # main("data/msmarco/passage/cn/qrels.dev.small-sorted-100.tsv", "dlm/runs/run.monot5.zh-45000.dev.tsv")

