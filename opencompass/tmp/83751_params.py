datasets = [
    [
        dict(
            abbr='nq',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.NQEvaluator'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt='Question: {question}?\nAnswer: ',
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/nq/',
            reader_cfg=dict(
                input_columns=[
                    'question',
                ],
                output_column='answer',
                train_split='test'),
            type='opencompass.datasets.NaturalQuestionDataset'),
    ],
]
models = [
    dict(
        abbr='llama-3-8b-instruct-ragga',
        batch_padding=True,
        batch_size=8,
        generation_kwargs=dict(
            do_sample=True,
            eos_token_id=[
                128001,
                128009,
            ],
            temperature=0.6,
            top_p=0.9),
        max_out_len=2048,
        meta_template=dict(
            begin='system',
            round=[
                dict(begin='user', end='<|eot_id|>', role='HUMAN'),
                dict(begin='', generate=True, role='BOT'),
            ]),
        model_kwargs=dict(device_map='auto', torch_dtype='torch.bfloat16'),
        path='/data/wxh/RAG/code/llama3-8b-instruct-ragga',
        run_cfg=dict(num_gpus=4),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='/data/wxh/RAG/code/llama3-8b-instruct-ragga',
        type='opencompass.models.myModel'),
]
work_dir = 'outputs/default/20240601_165202'
