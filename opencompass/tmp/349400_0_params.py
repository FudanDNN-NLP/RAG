datasets = [
    [
        dict(
            abbr='HotpotQA',
            eval_cfg=dict(
                evaluator=dict(type='opencompass.datasets.HotpotQAEvaluator'),
                pred_role='BOT'),
            infer_cfg=dict(
                inferencer=dict(
                    type='opencompass.openicl.icl_inferencer.GenInferencer'),
                prompt_template=dict(
                    template=dict(round=[
                        dict(
                            prompt='Question: {question}\nAnswer: ',
                            role='HUMAN'),
                    ]),
                    type=
                    'opencompass.openicl.icl_prompt_template.PromptTemplate'),
                retriever=dict(
                    type='opencompass.openicl.icl_retriever.ZeroRetriever')),
            path='./data/my_datasets/hotpotqa',
            reader_cfg=dict(
                context_column='context',
                input_columns='question',
                output_column='answer'),
            type='opencompass.datasets.HotpotQADataset'),
    ],
]
models = [
    dict(
        abbr='llama-3-8b-instruct',
        batch_padding=True,
        batch_size=12,
        max_out_len=80,
        model_kwargs=dict(device_map='auto', torch_dtype='torch.bfloat16'),
        path='/data/wxh/Meta-Llama-3-8B-Instruct',
        run_cfg=dict(num_gpus=4),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True),
        tokenizer_path='/data/wxh/Meta-Llama-3-8B-Instruct',
        type='opencompass.models.myModel'),
]
work_dir = 'outputs/zhangqi/20240605_195123'
