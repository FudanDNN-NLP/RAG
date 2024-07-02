import inspect
import os
import os.path as osp
from typing import List, Optional
import mmengine
import torch
from tqdm import tqdm
from opencompass.models.base import BaseModel
from opencompass.registry import ICL_INFERENCERS
from opencompass.utils import batched
from ..icl_prompt_template import PromptTemplate
from ..icl_retriever import BaseRetriever
from ..utils.logging import get_logger
from .icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import torch
import time

logger = get_logger(__name__)
import sys

default_retrieval_config = {
    "w_q": 0.3,  # 查询改写⽅法在最终检索结果中的权重, default: 0.3
    "w_d": 0.4,  # 伪⽂档⽣成⽅法在最终检索结果中的权重, default: 0.4
    "w_k": 0.3,  # 基于关键词的搜索⽅法在最终检索结果中的权重, default: 0.3
    "search_k": 10,  # search top_k
    "compression_ratio": 0.4,  # compression ratio, default: 0.4
    "top_k": 5,  # rerank top_k
    "Vector_Store": "milvus",  # Vector_Store: ["milvus"]
    # 上面的参数作为超参，暂时不变
    "with_retrieval_classification": True,
    "search_method": "hyde",  # search_method: ["hyde", "original", "hybrid"]
    "rerank_model": "MonoT5",  # rerank model: ["MonoT5", "TILDE", "MonoBERT", "RankLLaMA"]
    "compression_method": "longllmlingua",  # compression_method: ["longllmlingua", "recomp"]
    "repack_method": "sides",  # repack_method: ["sides", "compact", "compact_reverse"]
}  # total: 1+2+3+1+2+1=10


arguments = sys.argv  # 跳过脚本名称，获取剩余的参数
print("!!!!!!!!!!!!!!!!!!!!IGI")
print(arguments)
# ['run.py', './configs/myeval/eval_mymodel.py', '--debug', '--jt', '0', '--milvus', '0']
if "infer" in arguments[0]:
    import JointTest.JointRetrival2 as jt

    milvus_id = (int)(os.environ.get("milvus"))
    retrieval_config = default_retrieval_config.copy()
    retrieval_config["with_retrieval_classification"] = os.environ.get("classification") in ["true", "True"]
    retrieval_config["search_method"] = os.environ.get("search_method")
    retrieval_config["rerank_model"] = os.environ.get("rerank_model")
    retrieval_config["compression_method"] = os.environ.get("compression_method")
    retrieval_config["repack_method"] = os.environ.get("repack_method")
    print(f"!!!!!!!!!!!!!!!!!!!!jt:{retrieval_config},milvus:{milvus_id}")
    test_retrieval = jt.JointRetrieval(device="cuda", retrieval_config=retrieval_config, milvus_id=milvus_id)


def myretrieve(classification_query: str, query: str) -> str:
    Docs = test_retrieval.retrieval(classification_query, query)
    # print(Docs)
    if Docs.endswith("\n\n") == False:
        Docs += "\n"
    if Docs.endswith("\n\n") == False:
        Docs += "\n"
    return Docs


@ICL_INFERENCERS.register_module()
class GenInferencer(BaseInferencer):
    """Generation Inferencer class to directly evaluate by generation.

    Attributes:
        model (:obj:`BaseModelWrapper`, optional): The module to inference.
        max_seq_len (:obj:`int`, optional): Maximum number of tokenized words
            allowed by the LM.
        min_out_len (:obj:`int`, optional): Minimum number of generated tokens
            by the LM
        batch_size (:obj:`int`, optional): Batch size for the
            :obj:`DataLoader`.
        output_json_filepath (:obj:`str`, optional): File path for output
            `JSON` file.
        output_json_filename (:obj:`str`, optional): File name for output
            `JSON` file.
        gen_field_replace_token (:obj:`str`, optional): Used to replace the
            generation field token when generating prompts.
        save_every (:obj:`int`, optional): Save intermediate results every
            `save_every` iters. Defaults to 1.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the
            :obj:`model.generate()` method.
    """

    def __init__(
        self,
        model: BaseModel,
        max_out_len: int,
        stopping_criteria: List[str] = [],
        max_seq_len: Optional[int] = None,
        min_out_len: Optional[int] = None,
        batch_size: Optional[int] = 1,
        gen_field_replace_token: Optional[str] = "",
        output_json_filepath: Optional[str] = "./icl_inference_output",
        output_json_filename: Optional[str] = "predictions",
        save_every: Optional[int] = 1,
        **kwargs,
    ) -> None:
        super().__init__(
            model=model,
            max_seq_len=max_seq_len,
            batch_size=batch_size,
            output_json_filename=output_json_filename,
            output_json_filepath=output_json_filepath,
            **kwargs,
        )

        self.gen_field_replace_token = gen_field_replace_token
        self.max_out_len = max_out_len
        self.min_out_len = min_out_len
        self.stopping_criteria = stopping_criteria

        if self.model.is_api and save_every is None:
            save_every = 1
        self.save_every = save_every

    def inference(
        self,
        retriever: BaseRetriever,
        ice_template: Optional[PromptTemplate] = None,
        prompt_template: Optional[PromptTemplate] = None,
        output_json_filepath: Optional[str] = None,
        output_json_filename: Optional[str] = None,
    ) -> List:
        # 1. Preparation for output logs
        output_handler = GenInferencerOutputHandler()
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()

        # 3. Generate prompts for testing input
        prompt_list = self.get_generation_prompt_list_from_retriever_indices(
            ice_idx_list,
            retriever,
            self.gen_field_replace_token,
            max_seq_len=self.max_seq_len,
            ice_template=ice_template,
            prompt_template=prompt_template,
        )

        # 3.1 Fetch and zip prompt & gold answer if output column exists
        ds_reader = retriever.dataset_reader
        if ds_reader.output_column:
            gold_ans = ds_reader.dataset["test"][ds_reader.output_column]

            # prompt_list包含全部question和gold
            prompt_list = list(zip(prompt_list, gold_ans))

        # Create tmp json file for saving intermediate results and future
        # resuming
        index = 0
        tmp_json_filepath = os.path.join(output_json_filepath, "tmp_" + output_json_filename)
        if osp.exists(tmp_json_filepath):
            # TODO: move resume to output handler
            try:
                tmp_result_dict = mmengine.load(tmp_json_filepath)
            except Exception:
                pass
            else:
                output_handler.results_dict = tmp_result_dict
                index = len(tmp_result_dict)

        # 4. Wrap prompts with Dataloader
        dataloader = self.get_dataloader(prompt_list, self.batch_size)

        # 5. Inference for prompts in each batch
        logger.info("Starting inference process...")
        for datum in tqdm(dataloader, disable=not self.is_main_process):

            if ds_reader.output_column:
                entry, golds = list(zip(*datum))
            else:
                entry = datum
                golds = [None for _ in range(len(entry))]
            # 5-1. Inference with local model
            extra_gen_kwargs = {}
            sig = inspect.signature(self.model.generate)
            if "stopping_criteria" in sig.parameters:
                extra_gen_kwargs["stopping_criteria"] = self.stopping_criteria
            if "min_out_len" in sig.parameters:
                extra_gen_kwargs["min_out_len"] = self.min_out_len
            with torch.no_grad():

                # _________________________________________________________________________________________________________________
                contexts = []
                for idx, item in enumerate(entry):

                    if "question_stem" in ds_reader.input_columns:
                        query = ds_reader.dataset["test"]["question_stem"][index + idx]
                    elif "question" in ds_reader.input_columns:
                        query = ds_reader.dataset["test"]["question"][index + idx]
                    elif "claim" in ds_reader.input_columns:
                        query = ds_reader.dataset["test"]["claim"][index + idx]
                    elif "input" in ds_reader.input_columns:
                        query = ds_reader.dataset["test"]["input"][index + idx]
                    if "A" in ds_reader.input_columns:
                        A = ds_reader.dataset["test"]["A"][index + idx]
                        B = ds_reader.dataset["test"]["B"][index + idx]
                        C = ds_reader.dataset["test"]["C"][index + idx]
                        D = ds_reader.dataset["test"]["D"][index + idx]
                        query += f"{query}\nA. {A}\nB. {B}\nC. {C}\nD. {D}"

                    docs = myretrieve(entry[idx][1]["prompt"], query)
                    # documents="temp contexts"

                    # doc = ds_reader.dataset["test"]["doc"][index + idx]
                    # if doc == []:
                    #     docs = "\n\n"
                    # else:
                    #     # result = []

                    #     # # 对超长context进行截断
                    #     # for i in doc:
                    #     #     words = i.split()
                    #     #     if len(words) > 512:
                    #     #         truncated = " ".join(words[:512])
                    #     #     else:
                    #     #         truncated = i
                    #     #     result.append(truncated)

                    #     # dd = "\n\n".join(doc)
                    #     reranked_docs, _ = test_retrieval._rerank(
                    #         query=ds_reader.dataset["test"]["query"][index + idx],
                    #         model=test_retrieval.rerank_model,
                    #         expansion_model=test_retrieval.expansion_model,
                    #         tokenizer=test_retrieval.rerank_tokenizer,
                    #         bert_tokenizer=test_retrieval.bert_tokenizer,
                    #         docs=doc,
                    #         retrieval_config=test_retrieval.retrieval_config,
                    #     )

                    #     docs = test_retrieval._repack(ds_reader.dataset["test"]["query"][index + idx], reranked_docs[0])

                    #     docs, _ = test_retrieval._compress(
                    #         query=ds_reader.dataset["test"]["query"][index + idx],
                    #         model=test_retrieval.compressor_model,
                    #         tokenizer=test_retrieval.compressor_tokenizer,
                    #         docs=docs,
                    #         retrieval_config=test_retrieval.retrieval_config,
                    #     )
                    # if docs.endswith("\n\n") == False:
                    #     docs += "\n"
                    # if docs.endswith("\n\n") == False:
                    #     docs += "\n"

                    contexts.append(docs)
                    entry[idx][1]["prompt"] = "Background:" + docs + entry[idx][1]["prompt"]

                parsed_entries = self.model.parse_template(entry, mode="gen")
                results = self.model.generate_from_template(entry, max_out_len=self.max_out_len, **extra_gen_kwargs)
                generated = results

            num_return_sequences = getattr(self.model, "generation_kwargs", {}).get("num_return_sequences", 1)
            # 5-3. Save current output
            for prompt, prediction, gold, context in zip(
                parsed_entries,
                batched(generated, num_return_sequences),
                golds,
                contexts,
            ):
                if num_return_sequences == 1:
                    prediction = prediction[0]
                output_handler.save_results(prompt, prediction, index, gold=gold, context=context)
                index = index + 1

            # 5-4. Save intermediate results
            if self.save_every is not None and index % self.save_every == 0 and self.is_main_process:
                output_handler.write_to_json(output_json_filepath, "tmp_" + output_json_filename)

        # 6. Output
        if self.is_main_process:
            os.makedirs(output_json_filepath, exist_ok=True)
            output_handler.write_to_json(output_json_filepath, output_json_filename)
            if osp.exists(tmp_json_filepath):
                os.remove(tmp_json_filepath)

        return [sample["prediction"] for sample in output_handler.results_dict.values()]

    def get_generation_prompt_list_from_retriever_indices(
        self,
        ice_idx_list: List[List[int]],
        retriever: BaseRetriever,
        gen_field_replace_token: str,
        max_seq_len: Optional[int] = None,
        ice_template: Optional[PromptTemplate] = None,
        prompt_template: Optional[PromptTemplate] = None,
    ):
        prompt_list = []
        for idx, ice_idx in enumerate(ice_idx_list):
            ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
            prompt = retriever.generate_prompt_for_generate_task(
                idx,
                ice,
                gen_field_replace_token=gen_field_replace_token,
                ice_template=ice_template,
                prompt_template=prompt_template,
            )
            if max_seq_len is not None:
                prompt_token_num = self.model.get_token_len_from_template(prompt, mode="gen")
                while len(ice_idx) > 0 and prompt_token_num > max_seq_len:
                    ice_idx = ice_idx[:-1]
                    ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
                    prompt = retriever.generate_prompt_for_generate_task(
                        idx,
                        ice,
                        gen_field_replace_token=gen_field_replace_token,
                        ice_template=ice_template,
                        prompt_template=prompt_template,
                    )
                    prompt_token_num = self.model.get_token_len_from_template(prompt, mode="gen")
            prompt_list.append(prompt)
        return prompt_list
