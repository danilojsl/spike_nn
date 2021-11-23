from transformers import TFBertModel, BertTokenizer

MODEL_NAME = 'bert-base-cased'

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME).save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))
    model = TFBertModel.from_pretrained(MODEL_NAME).save_pretrained("./{}".format(MODEL_NAME), saved_model=True)
