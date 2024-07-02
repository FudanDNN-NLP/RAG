This guide explains how TILDEv2 can be used as a learned sparse index

### 1. Get JSONL file from TILDEv2 expansion/weights
You will need to refer to the main instructions and run the `expansion.py` tool followed by the `indexingv2.py` script. 
Assuming you have done those steps and have a directory with the TILDEv2 index in it (`/path/to/tildev2/index`) then:
Run the quantizer:
```
mkdir tildev2-jsonl
python3 quantize2json.py --input_path=/path/to/tildev2/index --output_file=tildev2-jsonl/tildev2.jsonl --quantize-bits=8
```

### 2. Build an Anserini Index over the jsonl file
You can split the jsonl file if you'd like to make indexing go faster (otherwise the threads option will not help).
Please see the instructions on the [Anserini page](https://github.com/castorini/anserini) for more options.
```
anserini/target/appassembler/bin/IndexCollection -collection JsonVectorCollection -generator DefaultLuceneDocumentGenerator -threads 18 -input tildev2-jsonl/ -index anserini-tildev2 -impact -optimize -pretokenized
```

### 3. Convert to CIFF
You then convert the Anserini index to the common index file format. [See the guide here.](https://github.com/osirrc/ciff)
```
ciff/target/appassembler/bin/ExportAnseriniLuceneIndex -index anserini-tildev2 -output tildev2.ciff -description "Anserini TILDEv2"
```

### 4. Run BP reordering
If you do not have a rust environment, you will need to install one. [See here](https://www.rust-lang.org/tools/install)
```
git clone https://github.com/JMMackenzie/enhanced-graph-bisection/
cd enhanced-graph-bisection
GAIN=approx_2 cargo build --release
cd ..
enhanced-graph-bisection/target/release/create-rgb -i tildev2.ciff -o bp-tildev2.ciff -m 256 --loggap
```

### 5. Convert CIFF to PISA canonical
Now we convert the reordered CIFF file to a PISA canonical index.
You can install the tool with cargo like: `cargo install ciff` or you can [look here for more information.](https://github.com/pisa-engine/ciff)
```
mkdir pisa-canonical
ciff2pisa --ciff-file bp-tildev2.ciff --output pisa-canonical/bp-tildev2
```

### 6. Index via PISA
Assuming you have [pisa cloned and built](https://github.com/pisa-engine/pisa/), execute the following commands.
```
mkdir pisa-index

# The inverted index
pisa/build/bin/compress_inverted_index --encoding block_simdbp --collection pisa-canonical/bp-tildev2 --output pisa-index/bp-tildev2.block_simdbp.idx

# The WAND data for skipping
pisa/build/bin/create_wand_data --collection pisa-canonical/bp-tildev2 --block-size 40 --scorer quantized --output pisa-index/bp-tildev2.fixed-40.bmw

# Lexicons
pisa/build/bin/lexicon build pisa-canonical/bp-tildev2.documents pisa-index/bp-tildev2.doclex
pisa/build/bin/lexicon build pisa-canonical/bp-tildev2.terms pisa-index/bp-tildev2.termlex
```

### 7. Get the queries
```
python3 dump_queries.py --query_path=data/queries/DL2019-queries.tsv | sed -e's/\t/ /' > DL2019-tilde.query
```

We also need to convert the BERT tokens to PISA index offsets; this is because PISA cannot currently parse BERT tokens :-(
```
awk -F" " 'NR==FNR{a[$1]=i++;next}{printf $1":"; for(i=2; i <= NF; ++i){printf " "a[$i]} print""}' pisa-canonical/bp-tildev2.terms DL2019-tilde.query > DL2019-tilde.pisa
```

### 8. Run some queries
Get a TREC run out:
```
pisa/build/bin/evaluate_queries -e block_simdbp -i pisa-index/bp-tildev2.block_simdbp.idx -w pisa-index/bp-tildev2.fixed-40.bmw -s quantized -k 1000 -q  DL2019-tilde.pisa -a maxscore --documents pisa-index/bp-tildev2.doclex > tilde-v2.run

trec_eval -m ndcg_cut.10 -m map data/qrels/2019qrels-pass.txt tilde-v2.run
map                   	all	0.4262
ndcg_cut_10           	all	0.6527

```

Do some latency measurement:
```
pisa/build/bin/queries -e block_simdbp -i pisa-index/bp-tildev2.block_simdbp.idx -w pisa-index/bp-tildev2.fixed-40.bmw -s quantized -k 1000 -q  DL2019-tilde.pisa -a maxscore
[2022-11-21 05:55:47.507] [stderr] [info] Loading index from pisa-index/bp-tildev2.block_simdbp.idx
[2022-11-21 05:55:47.856] [stderr] [info] Warming up posting lists
[2022-11-21 05:55:47.953] [stderr] [info] Performing block_simdbp queries
[2022-11-21 05:55:47.953] [stderr] [info] K: 1000
[2022-11-21 05:55:47.953] [stderr] [info] Query type: maxscore
[2022-11-21 05:55:47.953] [stderr] [info] Safe: false
[2022-11-21 05:55:56.337] [stderr] [info] ---- block_simdbp maxscore
[2022-11-21 05:55:56.337] [stderr] [info] Mean: 14005.7
[2022-11-21 05:55:56.337] [stderr] [info] 50% quantile: 9552
[2022-11-21 05:55:56.337] [stderr] [info] 90% quantile: 29059
[2022-11-21 05:55:56.337] [stderr] [info] 95% quantile: 42783
[2022-11-21 05:55:56.337] [stderr] [info] 99% quantile: 69769
[2022-11-21 05:55:56.337] [stderr] [info] Num. reruns: 0
{"type": "block_simdbp", "query": "maxscore", "avg": 14005.7, "q50": 9552, "q90": 29059, "q95": 42783, "q99": 69769}

```
