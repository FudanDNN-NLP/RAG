from transformers import PegasusForConditionalGeneration
# Need to download tokenizers_pegasus.py and other Python script from Fengshenbang-LM github repo in advance,
# or you can download tokenizers_pegasus.py and data_utils.py in https://huggingface.co/IDEA-CCNL/Randeng_Pegasus_523M/tree/main
# Strongly recommend you git clone the Fengshenbang-LM repo:
# 1. git clone https://github.com/IDEA-CCNL/Fengshenbang-LM
# 2. cd Fengshenbang-LM/fengshen/examples/pegasus/
# and then you will see the tokenizers_pegasus.py and data_utils.py which are needed by pegasus model
from tokenizers_pegasus import PegasusTokenizer

def pegasus_summary_zh(text):
        
        model = PegasusForConditionalGeneration.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")
        tokenizer = PegasusTokenizer.from_pretrained("IDEA-CCNL/Randeng-Pegasus-523M-Summary-Chinese")

        inputs = tokenizer(text, max_length=1024, return_tensors="pt")
        embedding_size = model.get_input_embeddings().weight.shape
        # print(embedding_size, len(tokenizer))

        model.resize_token_embeddings(50001)
        # Generate Summary
        summary_ids = model.generate(inputs["input_ids"])
        output = tokenizer.batch_decode(summary_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        return output


def main():
        return
if __name__=="__main__":
        
        main()
        
