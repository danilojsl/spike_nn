from transformers import BertTokenizer, TFBertModel, BertModel


def get_bert_tokenizer():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    return bert_tokenizer


def get_tf_bert_model():
    tf_bert_model = TFBertModel.from_pretrained("bert-base-cased")
    return tf_bert_model


def get_pt_bert_model():
    pt_bert_model = BertModel.from_pretrained('bert-base-cased')
    return pt_bert_model


if __name__ == '__main__':
    tokenizer = get_bert_tokenizer()
    tf_model = get_tf_bert_model()
    pt_model = get_pt_bert_model()

    text = "Replace me by any text you'd like."
    tf_encoded_input = tokenizer(text, return_tensors='tf')
    pt_encoded_input = tokenizer(text, return_tensors='pt')

    tf_output = tf_model(tf_encoded_input)
    pt_output = pt_model(**pt_encoded_input)

    print(tf_output)
    print(pt_model)
