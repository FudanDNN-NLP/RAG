from transformers import PegasusTokenizer, PegasusForConditionalGeneration, PegasusModel, PegasusConfig
import datasets
import time
import json
import pandas as pd

def pegasus_summary(text):
    
    model = PegasusForConditionalGeneration.from_pretrained("sshleifer/distill-pegasus-xsum-16-4")
    tokenizer = PegasusTokenizer.from_pretrained("sshleifer/distill-pegasus-xsum-16-4")
    
    return generate_for_sample(text,model,tokenizer) # {'summary':[''],'gold':''}

def generate_for_sample(sample, model , tokenizer):
        """
        Returns decoded summary (code snippets from the docs)
        kwargs are passed on to the model's generate function
        """
        inputs = tokenizer(sample, truncation=True, max_length=1024, return_tensors='pt')
        summary_ids = model.generate(inputs['input_ids'])
        return [tokenizer.decode(g, 
                                skip_special_tokens=True, 
                                clean_up_tokenization_spaces=False) for g in summary_ids]


def main():
    start = time.time()

    data = pd.read_json("testfile.json")


    model = PegasusForConditionalGeneration.from_pretrained("sshleifer/distill-pegasus-xsum-16-4")
    tokenizer = PegasusTokenizer.from_pretrained("sshleifer/distill-pegasus-xsum-16-4")
     
    
    # Download data samples
    # data = datasets.load_dataset("xsum", split="validation[:10]")

    # # Pick two examples
    # text2summarize_1 = data["document"][0]
    # text2summarize_2 = data["document"][3]


    # print(text2summarize_1) 
    # print(text2summarize_2)

   

    print("Summaries generated with default parameters:")

    output = []
    
    for i in range(5): 
         
         summary_p = generate_for_sample(data.loc[i,'passages'],model,tokenizer)
         output.append({'summary':summary_p,'gold':data.loc[i,'summary']})


    # with open("output.json","a+") as ot:
    #     ot.write(json.dumps(output))
    print(output[0]['summary'])



    # print("Some default parameter values: ", "num_beams={}, do_sample={}, top_k={}, top_p={}".
    #     format(model.config.num_beams, model.config.do_sample, model.config.top_k, model.config.top_p))

    # print("Summaries generated with custom parameter values:")
    # summary_1 = generate_for_sample(prompt, num_beams=4)

    # print("summary_1: {}".format(prompt))

    end = time.time()

    print(end-start)

if __name__=="__main__" : 
   
    main()
