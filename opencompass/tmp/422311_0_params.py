datasets = [
    [
        dict(
            abbr='PubHealth',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.pubhealthEvaluator'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt=
                            "Determine whether the given background supports or refutes the claim, and respond with 'SUPPORT' or 'REFUTE'. Do not generate any other content.\nClaim: {claim}\nAnswer: ",
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            reader_cfg=dict(input_columns=[
                'claim',
            ], output_column='label'),
            type='opencompass.datasets.pubhealthDataset'),
    ],
]
models = [
    dict(
        abbr='llama-3-8b-ragga-disturb',
        batch_padding=True,
        batch_size=1,
        generation_kwargs=dict(
            do_sample=False, eos_token_id=[
                128001,
                128009,
            ]),
        max_out_len=50,
        meta_template=dict(
            begin='<|start_header_id|>system<|end_header_id|>\n\n<|eot_id|>',
            round=[
                dict(
                    begin='<|start_header_id|>user<|end_header_id|>\n\n',
                    end='<|eot_id|>',
                    role='HUMAN'),
                dict(
                    begin='<|start_header_id|>assistant<|end_header_id|>\n\n',
                    generate=True,
                    role='BOT'),
            ]),
        model_kwargs=dict(device_map='auto', torch_dtype='torch.bfloat16'),
        path='/data/wxh/RAG/code/llama3-8b-instruct-ragga-disturb',
        run_cfg=dict(num_gpus=1),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='/data/wxh/RAG/code/llama3-8b-instruct-ragga-disturb',
        type='opencompass.models.myModel'),
]
work_dir = 'outputs/t_hh_mt_r_c/eval_pubhealth/20240613_132626'
