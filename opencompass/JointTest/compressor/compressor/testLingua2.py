from llmlingua import PromptCompressor
import time
import json
import pandas as pd
start_time = time.time()

data = pd.read_json("test.json")


llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)
llm_lingua = PromptCompressor(
    model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
    use_llmlingua2=True, # Whether to use llmlingua-2
)
output = []
for i in range(5):
    
   summary_p = llm_lingua.compress_prompt(data.loc[i,'passages'], rate=0.025
                                               , force_tokens = ['\n', '?'])
   output.append({'summary':summary_p,'gold':data.loc[i,'summary']})


end_time = time.time()

total_time = end_time - start_time
print("latency",total_time)

with open("output.json","a+") as ot:
    ot.write(json.dumps(output))



