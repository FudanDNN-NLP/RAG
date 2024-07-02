import random


def shuffle(input_file, output_file, doc_cnt):
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    first_column, second_column, third_column = [], [], []
    shuffled_lines = []
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        first_column.append(parts[0])
        second_column.append(parts[1])
        third_column.append(parts[2])
        if (i + 1) % doc_cnt == 0:
            random.shuffle(second_column)
            for j in range(doc_cnt):
                shuffled_lines.append(first_column[j] + '\t' + second_column[j] + '\t' + third_column[j] + '\n')
            first_column, second_column, third_column = [], [], []

    with open(output_file, 'w', encoding='utf-8') as f:
        f.writelines(shuffled_lines)
    print("Done! Shuffled file saved to {}".format(output_file))

if __name__ == '__main__':
    # shuffle('../data/msmarco_doc_ans_small/fh/run.dev.small.tsv', '../data/msmarco_doc_ans_small/fh/run.dev.small_shuffled.tsv', 1000)
    # shuffle('../data/msmarco_ans_small/run.dev.small.tsv', '../data/msmarco_ans_small/run.dev.small_shuffled.tsv', 1000)
    shuffle('../data/msmarco/passage/dev-6980/run.dev.small.tsv', '../data/msmarco/passage/dev-6980/run.dev.small_shuffled.tsv', 1000)