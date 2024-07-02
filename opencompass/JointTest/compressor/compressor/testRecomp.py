
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import ast
import time
import re
import datasets
# import evaluate
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from filelock import FileLock
import pandas as pd
import transformers



from transformers import ( # type: ignore
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode, send_example_telemetry
from transformers.utils.versions import require_version

from functools import partial
from datetime import datetime

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.30.0.dev0")

# require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: Optional[str] = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": (
                "Will use the token generated when running `huggingface-cli login` (necessary to use this script "
                "with private models)."
            )
        },
    )
    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
                "the model's position embeddings."
            )
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    lang: Optional[str] = field(default=None, metadata={"help": "Language id for summarization."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={"help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "An optional input evaluation data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on (a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": (
                "The maximum total input sequence length after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": (
                "The maximum total sequence length for target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded."
            )
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "The maximum total sequence length for validation target text after tokenization. Sequences longer "
                "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
                "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
                "during ``evaluate`` and ``predict``."
            )
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to pad all samples to model maximum sentence length. "
                "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
                "efficient on GPU but very bad for TPU."
            )
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of training examples to this "
                "value if set."
            )
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
                "value if set."
            )
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "For debugging purposes or quicker training, truncate the number of prediction examples to this "
                "value if set."
            )
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
                "which is used during ``evaluate`` and ``predict``."
            )
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The token to force as the first generated token after the decoder_start_token_id."
                "Useful for multilingual models like mBART where the first generated token"
                "needs to be the target language token (Usually it is the target language token)"
            )
        },
    )

    def __post_init__(self):
        if (
            self.dataset_name is None
            and self.train_file is None
            and self.validation_file is None
            and self.test_file is None
        ):
            raise ValueError("Need either a dataset name or a training, validation, or test file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in ["csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in ["csv", "json"], "`validation_file` should be a csv or a json file."
            if self.test_file is not None:
                extension = self.test_file.split(".")[-1]
                assert extension in ["csv", "json"], "`test_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length

summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
    "multi_news": ("document", "summary"),
}

def count_words(text):
    # 移除文本中的非字母数字字符，并将文本转换为小写
    text = re.sub(r'\W+', ' ', text.lower())
    # 将文本分割成单词，并返回单词的数量
    return text.split().count(' ')

def recomp_summary(query,docs,compression_ratio):
    
    # 中文文本长度计算？
    
    
    # 英文文本长度计算？
    max_target_length = compression_ratio*count_words(docs)
    
    return recomp_main(query,docs,512)

def recomp_main(query,docs,max_target_length = 512):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.
    
    sys.argv = sys.argv + ['--model_name_or_path', 'fangyuan/nq_abstractive_compressor','--output_dir', 'outputs/','--test_file', 'testfile.json']
    
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    # if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
    #     # If we pass only one argument to the script and it's the path to a json file,
    #     # let's parse it to get our arguments.
    #     model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        
    # else:
    #     # 执行
    #     model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    #     training_args.do_predict = True
    #     training_args.do_train = False
    #     training_args.do_eval = False
    #     training_args.output_dir = "outputs/"
    #     training_args.predict_with_generate = True
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.do_predict = True
    training_args.do_train = False
    training_args.do_eval = False
    training_args.output_dir = "outputs/"
    training_args.predict_with_generate = True
    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry("run_summarization", model_args, data_args)

    # Detecting last checkpoint.
    # last_checkpoint = None
    # overwrite
    output_dir = '{}-{}-{}'.format("outputs/", "fangyuan/nq_abstractive_compressor".replace("/", "-"),
                                        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    print("Output dir: {}".format(output_dir))
    

    # Set seed before initializing model.
    # set_seed(training_args.seed)

    # 创建数据集
    
    
    
    
    
    # data_files = {}
    # data_files["test"] = "testfile.json"
    # extension = "testfile.json".split(".")[-1]
    # raw_datasets = load_dataset(
        
    #     extension,
    #     data_files = data_files,
    #     # cache_dir=model_args.cache_dir,
    #     use_auth_token= None,
    # )
    

    
    data_dict = {
        
            "question": [query],
            "passages":[docs]
        
    }
    df = pd.DataFrame(data_dict)
    dt = Dataset.from_pandas(df)
    raw_datasets = DatasetDict({"test": dt})
    

    
    
    # ['gold_answers', 'summary', 'passages', 'question']
    
    config = AutoConfig.from_pretrained(
        "fangyuan/nq_abstractive_compressor",
        # cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token= None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "fangyuan/nq_abstractive_compressor",
        # cache_dir=model_args.cache_dir,
        use_fast= True,
        revision="main",
        use_auth_token= None,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        "fangyuan/nq_abstractive_compressor",
        from_tf=bool(".ckpt" in "fangyuan/nq_abstractive_compressor"),
        config=config,
        # cache_dir=model_args.cache_dir,
        revision="main",
        use_auth_token= None,
    )

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) > embedding_size:
        model.resize_token_embeddings(len(tokenizer))

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[None]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(None)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")
    
    if True:
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        column_names = raw_datasets["test"].column_names
        
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Temporarily set max_target_length for training.
    max_target_length = max_target_length
    padding = False

    def preprocess_summary_function(split, examples):
        # remove pairs where at least one record is None
        inputs, questions= [], []
        for i in range(len(examples['question'])):
            input_txt = "Question: {}\n Document: {}\n Summary: ".format(
                examples['question'][i],
                examples['passages'][i],
            )
            inputs.append(input_txt)
            questions.append(examples['question'][i])

        inputs = ["" + inp for inp in inputs]
        # print(inputs)
        model_inputs = tokenizer(inputs, max_length=2048, padding=padding, truncation=True)
        model_inputs["question"] = questions
        
        return model_inputs



    
    max_target_length = max_target_length
    predict_dataset = raw_datasets["test"]
    
    
    
    predict_dataset = predict_dataset.map(
        partial(preprocess_summary_function, 'test'),
        batched=True,
        num_proc=None,
        remove_columns=column_names,
        load_from_cache_file=not False,
        desc="Running tokenizer on prediction dataset",
    )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    # metric = evaluate.load("rouge")
    from rouge import Rouge
    rouge = Rouge()

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        # Replace -100s used for padding as we can't decode them
        preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        sample_idx = np.random.randint(len(decoded_preds))
        print(">> Prediction: {}".format(decoded_preds[sample_idx]))
        print(">> Reference: {}".format(decoded_labels[sample_idx]))
        # map empty label
        placeholder = "This passage doesn't contain relevant information to the question."
        decoded_preds = [pred if len(pred) > 1 else placeholder for pred in decoded_preds]
        result = {}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # Override the decoding parameters of Seq2SeqTrainer
    training_args.generation_max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    training_args.generation_num_beams = (
        data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    )
    
    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args = training_args,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
     
    # Evaluation
    results = {}
    
    predict_results = trainer.predict(predict_dataset, metric_key_prefix="predict")

    if trainer.is_world_process_zero():
            
            predictions = predict_results.predictions
            predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
            predictions = tokenizer.batch_decode(
                predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = os.path.join("outputs/", "generated_predictions.txt")
            with open(output_prediction_file, "w") as writer:
                writer.write("\n".join(predictions))
    
    return predictions


def _mp_fn(index):
    # For xla_spawn (TPUs)
    recomp_main("","",512)


if __name__ == "__main__":
    question = "What Old English poem commemorates the capturing of the five main towns of Danish Mercia?"
    passages = "the correct date added by another hand. Frank Stenton comments that the poem \"is overloaded with cliches\", but also packs a lot of historical information, recording how the conquest of Mercia by King Edmund liberated, in 942, the people of the Five Boroughs (Leicester, Lincoln, Derby, Nottingham, Stamford) from the Norsemen under Olaf Guthfrithson and Amla\u00edb Cuar\u00e1n. These people were not English\u2014rather, they were Danes, who by this time considered themselves so English that they preferred King Edmund over their Norse overlords who had invaded their territory from Viking York. According to Sarah Foot, these \"anglicised\" Danes, liberated by Edmund,\nmust thus have been Christian as well, and the poem aids in the construction of an English identity out of different ethnic groups united in their opposition to outside, pagan forces. Capture of the Five Boroughs \"Capture of the Five Boroughs\" (also \"Redemption of the Five Boroughs\") is an Old English chronicle poem that commemorates the capture by King Edmund I of the so-called Five Boroughs of the Danelaw in 942. The seven-line long poem is one of the five so-called \"chronicle poems\" found in the Anglo-Saxon Chronicle; it is preceded by \"The Battle of Brunanburh\" (937) and followed by\nthat of Leofric, Earl of Mercia, and it was to form a formal administrative unit long into the future. Five Boroughs of the Danelaw The Five Boroughs or The Five Boroughs of the Danelaw (Old Norse: \"Fimm Borginn\") were the five main towns of Danish Mercia (what is now the East Midlands). These were Derby, Leicester, Lincoln, Nottingham and Stamford. The first four later became county towns. Viking raids began in England in the late 8th century, and were largely of the 'hit and run' sort. However, in 865 various Viking armies combined and landed in East Anglia, not to\nCapture of the Five Boroughs \"Capture of the Five Boroughs\" (also \"Redemption of the Five Boroughs\") is an Old English chronicle poem that commemorates the capture by King Edmund I of the so-called Five Boroughs of the Danelaw in 942. The seven-line long poem is one of the five so-called \"chronicle poems\" found in the Anglo-Saxon Chronicle; it is preceded by \"The Battle of Brunanburh\" (937) and followed by the two poems on King Edgar. In the Parker MS, the text of \"Brunanburh\" is written by the same scribe as \"Capture\", which starts on the line for 941 but has\nFive Boroughs of the Danelaw The Five Boroughs or The Five Boroughs of the Danelaw (Old Norse: \"Fimm Borginn\") were the five main towns of Danish Mercia (what is now the East Midlands). These were Derby, Leicester, Lincoln, Nottingham and Stamford. The first four later became county towns. Viking raids began in England in the late 8th century, and were largely of the 'hit and run' sort. However, in 865 various Viking armies combined and landed in East Anglia, not to raid but to conquer the four Anglo-Saxon kingdoms of England. The annals described the combined force as the Great"
    
    print(recomp_summary(question,passages,512))
