from opencompass.models import HuggingFaceCausalLM, HuggingFacewithChatTemplate
import torch


models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="llama-3-8b-instruct-ragga",
        path="/data/wxh/RAG/code/llama3-8b-instruct-ragga",
        model_kwargs=dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ),
        tokenizer_path="/data/wxh/RAG/code/llama3-8b-instruct-ragga",
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            do_sample=True,
            temperature=0.6,
            top_p=0.9,
            eos_token_id=[128001,128009],
        ),
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=4),
        batch_padding=True,
        meta_template=dict(
            begin="system",
            round=[
                dict(
                    role="HUMAN", begin="user",end="<|eot_id|>"
                ),  # begin and end can be a list of strings or integers.
                dict(role="BOT", begin="", generate=True),
            ],
        ),
    )
]
