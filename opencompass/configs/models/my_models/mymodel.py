from opencompass.models import myModel, HuggingFaceCausalLM
import torch


models = [
    dict(
        type=myModel,
        abbr="llama-3-8b-ragga-disturb",
        path="/data/wxh/RAG/code/llama3-8b-instruct-ragga-disturb",
        model_kwargs=dict(
            torch_dtype=torch.bfloat16,
            device_map="auto",
        ),
        tokenizer_path="/data/wxh/RAG/code/llama3-8b-instruct-ragga-disturb",
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            do_sample=False,
            eos_token_id=[128001, 128009],
        ),
        max_out_len=50,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        batch_padding=True,
        meta_template=dict(
            begin="<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>",
            round=[
                dict(
                    role="HUMAN",
                    begin="<|start_header_id|>user<|end_header_id|>\n\n",
                    end="<|eot_id|>",
                ),  
                dict(
                    role="BOT",
                    begin="<|start_header_id|>assistant<|end_header_id|>\n\n",
                    generate=True,
                ),
            ],
        ),
    )
]
