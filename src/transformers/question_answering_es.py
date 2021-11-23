from transformers import BertForQuestionAnswering, AutoTokenizer
from transformers import pipeline

model_name = 'mrm8488/distill-bert-base-spanish-wwm-cased-finetuned-spa-squad2-es'
local_model_path = "/home/danilo/tmp/models/bert"
local_tokenizer_model_path = "/home/danilo/tmp/models/bert_tokenizer"


def nlp_pipeline():
    remote_model = BertForQuestionAnswering.from_pretrained(model_name)
    model.save_pretrained(local_model_path)
    remote_tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(local_tokenizer_model_path)

    return remote_model, remote_tokenizer


def local_nlp_pipeline():
    local_model = BertForQuestionAnswering.from_pretrained(local_model_path)
    local_tokenizer_model = AutoTokenizer.from_pretrained(local_tokenizer_model_path)

    return local_model, local_tokenizer_model


if __name__ == '__main__':
    # model, tokenizer = nlp_pipeline()
    model, tokenizer = local_nlp_pipeline()

    nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    context = "Buenos dias, mi nombre es Danilo y quiero tener sexo salvaje con mi esposa Yadira"

    question = 'Que quiere Danilo?'

    result = nlp({'question': question, 'context': context})
    print(result)
