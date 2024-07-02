import numpy as np
import torch
import transformers
import os
from typing import Dict, List, Optional, Union

from opencompass.models.base import BaseModel
from opencompass.registry import MODELS
from opencompass.utils.logging import get_logger
from opencompass.utils.prompt import PromptList

PromptType = Union[PromptList, str]

@MODELS.register_module()
class myModel(BaseModel):
    """Model wrapper around HuggingFace models.

    Args:
        path (str): The name or path to HuggingFace's model.
        hf_cache_dir: Set the cache dir to HF model cache dir. If None, it will
            use the env variable HF_MODEL_HUB. Defaults to None.
        max_seq_len (int): The maximum length of the input sequence. Defaults
            to 2048.
        tokenizer_path (str): The path to the tokenizer. Defaults to None.
        tokenizer_kwargs (dict): Keyword arguments for the tokenizer.
            Defaults to {}.
        peft_path (str, optional): The name or path to the HuggingFace's PEFT
            model. If None, the original model will not be converted to PEFT.
            Defaults to None.
        tokenizer_only (bool): If True, only the tokenizer will be initialized.
            Defaults to False.
        model_kwargs (dict): Keyword arguments for the model, used in loader.
            Defaults to dict(device_map='auto').
        meta_template (Dict, optional): The model's meta prompt
            template if needed, in case the requirement of injecting or
            wrapping of any meta instructions.
        extract_pred_after_decode (bool): Whether to extract the prediction
            string from the decoded output string, instead of extract the
            prediction tokens before decoding. Defaults to False.
        batch_padding (bool): If False, inference with be performed in for-loop
            without batch padding.
        pad_token_id (int): The id of the padding token. Defaults to None. Use
            (#vocab + pad_token_id) if get negative value.
        mode (str, optional): The method of input truncation when input length
            exceeds max_seq_len. 'mid' represents the part of input to
            truncate. Defaults to 'none'.
        use_fastchat_template (str, optional): Whether to use fastchat to get
            the conversation template. If True, fastchat needs to be
            implemented first. Defaults to False.
        end_str (str, optional): Whether to trim generated strings with end_str
            if the model has special ending strings that are not handled well.
            Defaults to None.

    Note:
        About ``extract_pred_after_decode``: Commonly, we should extract the
        the prediction tokens before decoding. But for some tokenizers using
        ``sentencepiece``, like LLaMA,  this behavior may change the number of
        whitespaces, which is harmful for Python programming tasks.
    """

    def __init__(
        self,
        path: str,
        hf_cache_dir: Optional[str] = None,
        max_seq_len: int = 2048,
        tokenizer_path: Optional[str] = None,
        tokenizer_kwargs: dict = dict(),
        peft_path: Optional[str] = None,
        tokenizer_only: bool = False,
        model_kwargs: dict = dict(device_map="auto"),
        generation_kwargs: dict = dict(),
        meta_template: Optional[Dict] = None,
        extract_pred_after_decode: bool = False,
        batch_padding: bool = False,
        pad_token_id: Optional[int] = None,
        mode: str = "none",
        use_fastchat_template: bool = False,
        end_str: Optional[str] = None,
    ):
        super().__init__(
            path=path,
            max_seq_len=max_seq_len,
            tokenizer_only=tokenizer_only,
            meta_template=meta_template,
        )
        if hf_cache_dir is None:
            hf_cache_dir = os.getenv("HF_MODEL_HUB", None)
        self.logger = get_logger()
        self.pad_token_id = pad_token_id
        assert mode in ["none", "mid"]
        self.mode = mode
        self._load_tokenizer(
            path=path, tokenizer_path=tokenizer_path, tokenizer_kwargs=tokenizer_kwargs
        )
        self.batch_padding = batch_padding
        self.extract_pred_after_decode = extract_pred_after_decode
        if not tokenizer_only:
            self._load_model(path=path, model_kwargs=model_kwargs, peft_path=peft_path)
        self.generation_kwargs = generation_kwargs
        self.use_fastchat_template = use_fastchat_template
        self.end_str = end_str

    def _load_tokenizer(
        self, path: str, tokenizer_path: Optional[str], tokenizer_kwargs: dict
    ):
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path if tokenizer_path else path, **tokenizer_kwargs
        )

        # A patch for some models without pad_token_id
        if self.pad_token_id is not None:
            if self.pad_token_id < 0:
                self.pad_token_id += self.tokenizer.vocab_size
            if self.tokenizer.pad_token_id is None:
                self.logger.debug(f"Using {self.pad_token_id} as pad_token_id")
            elif self.tokenizer.pad_token_id != self.pad_token_id:
                self.logger.warning(
                    "pad_token_id is not consistent with the tokenizer. Using "
                    f"{self.pad_token_id} as pad_token_id"
                )
            self.tokenizer.pad_token_id = self.pad_token_id
        elif self.tokenizer.pad_token_id is None:
            self.logger.warning("pad_token_id is not set for the tokenizer.")
            if self.tokenizer.eos_token is not None:
                self.logger.warning(
                    f"Using eos_token_id {self.tokenizer.eos_token} " "as pad_token_id."
                )
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                from transformers.generation import GenerationConfig

                gcfg = GenerationConfig.from_pretrained(path)

                if gcfg.pad_token_id is not None:
                    self.logger.warning(
                        f"Using pad_token_id {gcfg.pad_token_id} " "as pad_token_id."
                    )
                    self.tokenizer.pad_token_id = gcfg.pad_token_id
                else:
                    raise ValueError(
                        "pad_token_id is not set for this tokenizer. Try to "
                        "set pad_token_id via passing "
                        "`pad_token_id={PAD_TOKEN_ID}` in model_cfg."
                    )

    def _set_model_kwargs_torch_dtype(self, model_kwargs):
        if "torch_dtype" not in model_kwargs:
            torch_dtype = torch.float16
        else:
            torch_dtype = {
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
                "torch.float": torch.float,
                "auto": "auto",
                "None": None,
            }.get(model_kwargs["torch_dtype"])
        self.logger.debug(f"HF using torch_dtype: {torch_dtype}")
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype

    def _load_model(
        self, path: str, model_kwargs: dict, peft_path: Optional[str] = None
    ):
        from transformers import AutoModel, AutoModelForCausalLM

        self._set_model_kwargs_torch_dtype(model_kwargs)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
        except ValueError:
            self.model = AutoModel.from_pretrained(path, **model_kwargs)

        self.model.eval()
        self.model.generation_config.do_sample = False

    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        """Generate results given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.
            min_out_len (Optional[int]): The minimum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        generation_kwargs = kwargs.copy()
        generation_kwargs.update(self.generation_kwargs)
        if self.batch_padding and len(inputs) > 1:
            return self._batch_generate(
                inputs=inputs,
                max_out_len=max_out_len,
                min_out_len=min_out_len,
                stopping_criteria=stopping_criteria,
                **generation_kwargs,
            )
        else:
            return sum(
                (
                    self._single_generate(
                        inputs=[input_],
                        max_out_len=max_out_len,
                        min_out_len=min_out_len,
                        stopping_criteria=stopping_criteria,
                        **generation_kwargs,
                    )
                    for input_ in inputs
                ),
                [],
            )

    def _batch_generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        """Support for batch prompts inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        # if self.extract_pred_after_decode:
        #     prompt_lens = [len(input_) for input_ in inputs]

        # step-1: tokenize the input with batch_encode_plus
        tokens = self.tokenizer.batch_encode_plus(
            inputs, padding=True, truncation=True, max_length=self.max_seq_len
        )
        tokens = {
            k: torch.tensor(np.array(tokens[k]), device=self.model.device)
            for k in tokens
            if k in ["input_ids", "attention_mask"]
        }

        # self.logger.warning(f"!!!!!!{tokens}")

        if min_out_len is not None:
            kwargs["min_new_tokens"] = min_out_len

        # step-2: conduct model forward to generate output
        outputs = self.model.generate(**tokens, max_new_tokens=max_out_len, **kwargs)

        if not self.extract_pred_after_decode:
            # outputs = outputs[:, tokens['input_ids'].shape[1]:]
            outputs = outputs.sequences[
                :, tokens["input_ids"].shape[1] :
            ]  # 5.31 1:00修改 为了解决TypeError: tuple indices must be integers or slices, not tuple问题

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [token[len_:] for token, len_ in zip(decodeds, prompt_lens)]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]

        return decodeds

    def _single_generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ) -> List[str]:
        """Support for single prompt inference.

        Args:
            inputs (List[str]): A list of strings.
            max_out_len (int): The maximum length of the output.

        Returns:
            List[str]: A list of generated strings.
        """
        # if self.extract_pred_after_decode:
        #     prompt_lens = [len(input_) for input_ in inputs]

        input_ids = self.tokenizer(
            inputs, truncation=True, max_length=self.max_seq_len - max_out_len
        )["input_ids"]
        input_ids = torch.tensor(input_ids, device=self.model.device)

        if min_out_len is not None:
            kwargs["min_new_tokens"] = min_out_len

        # To accommodate the PeftModel, parameters should be passed in
        # key-value format for generate.
        outputs = self.model.generate(
            input_ids=input_ids, max_new_tokens=max_out_len, **kwargs
        )

        if not self.extract_pred_after_decode:
            # outputs = outputs[:, input_ids.shape[1] :]
            outputs = outputs.sequences[:, input_ids.shape[1] :]         # 解决bug

        decodeds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if self.extract_pred_after_decode:
            decodeds = [token[len_:] for token, len_ in zip(decodeds, prompt_lens)]

        if self.end_str:
            decodeds = [token.split(self.end_str)[0] for token in decodeds]

        return decodeds

    def get_logits(self, inputs: List[str]):

        if self.batch_padding and len(inputs) > 1:
            # batch inference
            tokens = self.tokenizer(
                inputs, padding=True, truncation=True, max_length=self.max_seq_len
            )

            tokens = {
                k: torch.tensor(np.array(tokens[k]), device=self.model.device)
                for k in tokens
                if k in ["input_ids", "attention_mask"]
            }
            outputs = self.model(**tokens)

        else:
            input_ids = self.tokenizer(
                inputs, padding=False, truncation=True, max_length=self.max_seq_len
            )["input_ids"]
            input_ids = torch.tensor(input_ids, device=self.model.device)
            tokens = {"input_ids": input_ids}

            outputs = self.model(input_ids)
        return outputs[0], {"tokens": tokens}

    def get_ppl(
        self, inputs: List[str], mask_length: Optional[List[int]] = None
    ) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_ppl(inputs, mask_length=mask_length)
        else:
            return np.concatenate(
                [
                    self._get_ppl(inputs=[text], mask_length=mask_length)
                    for text in inputs
                ]
            )

    def _get_ppl(
        self, inputs: List[str], mask_length: Optional[List[int]] = None
    ) -> List[float]:
        """Get perplexity scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of perplexity scores.
        """

        outputs, inputs = self.get_logits(inputs)
        shift_logits = outputs[..., :-1, :].contiguous().float()

        shift_labels = inputs["tokens"]["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(
            reduction="none", ignore_index=self.tokenizer.pad_token_id
        )
        loss = loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)
        ).view(shift_labels.size())

        if mask_length is not None:
            mask = torch.zeros_like(shift_labels)  # [batch,seqlen]
            for i in range(len(mask)):
                for j in range(mask_length[i] - 1, len(mask[i])):
                    mask[i][j] = 1
            loss = loss * mask

        lens = (
            (inputs["tokens"]["input_ids"] != self.tokenizer.pad_token_id)
            .sum(-1)
            .cpu()
            .numpy()
        )
        if mask_length is not None:
            lens -= np.array(mask_length)
        ce_loss = loss.float().sum(-1).cpu().detach().numpy() / lens
        return ce_loss

    def get_loglikelihood(
        self,
        inputs: List[str],
        conts: List[str],
        mask_length: Optional[List[int]] = None,
    ) -> List[float]:
        """Get loglikelihood scores given a list of inputs.

        Args:
            inputs (List[str]): A list of strings.
            conts (List[str]): A list of strings: slices after the space.
            NOT SUPPORT mask_length YET!
            mask_length (Optional[List[int]]): A list of mask lengths. If
                provided, the perplexity scores will be calculated with the
                first mask_length[i] tokens masked out. It's okay to skip
                its implementation if advanced features in PPLInfernecer is
                not needed.

        Returns:
            List[float]: A list of loglikelihood scores.
        """
        assert mask_length is None, "Not support mask_length yet."
        if self.batch_padding and len(inputs) > 1:
            assert self.tokenizer.pad_token
            return self._get_loglikelihood(inputs, conts)
        else:
            return np.concatenate(
                [
                    self._get_loglikelihood(inputs=[inputs[idx]], conts=[conts[idx]])
                    for idx in range(len(inputs))
                ]
            )

    def _get_loglikelihood(self, inputs: str, conts: str) -> float:
        """Get loglikelihood scores given input string and continuation string.

        Args:
            inputs (str): string.
            conts (str): strings: slices after the space.
        Returns:
            float: loglikelihood scores.
        """
        input_tokenizer_out = self.tokenizer(
            inputs,
            padding=True,
            truncation=False,
            return_length=True,
            return_tensors="pt",
        ).to(self.model.device)

        input_ids = input_tokenizer_out["input_ids"][:, : self.max_seq_len]
        input_length = input_tokenizer_out["length"]
        context_ids = [
            self.tokenizer(
                inputs[i].replace(conts[i], ""),
                padding=False,
                truncation=True,
                max_length=self.max_seq_len,
            )["input_ids"]
            for i in range(len(inputs))
        ]
        # forward
        outputs = self.model(input_ids)["logits"]
        outputs = torch.nn.functional.log_softmax(outputs, dim=-1)
        # calculate loglikelihood
        answer = np.zeros(len(inputs))
        for i in range(len(inputs)):
            if self.tokenizer.padding_side == "right":
                cont_ids = input_ids[i, len(context_ids[i]) : input_length[i]]
                logits = outputs[
                    i, len(context_ids[i]) - 1 : input_length[i] - 1, :
                ]  # noqa
            else:
                cont_ids = input_ids[i, len(context_ids[i]) - input_length[i] :]
                logits = outputs[i, len(context_ids[i]) - input_length[i] - 1 : -1]
            # Reducing the dimension will lead to a wrong outcome
            logits_gather = torch.gather(
                logits.unsqueeze(0), 2, cont_ids.unsqueeze(0).unsqueeze(-1)
            )  # [1, seq]
            # Answer: sum the likelihood of each token in continuation
            answer[i] = float(logits_gather.detach().cpu().sum())
        return answer

    def get_token_len(self, prompt: str) -> int:
        """Get lengths of the tokenized strings.

        Args:
            prompt (str): Input string.

        Returns:
            int: Length of the input tokens
        """
        return len(self.tokenizer.encode(prompt))


if __name__ == "__main__":
    model = myModel(pkg_root="zzzzz")
    model.generate(None, None)
