import tensorflow as tf
import requests
import json
from transformers import TFAutoModelForQuestionAnswering
from transformers import AutoTokenizer
# Attempt to save transformers models to as TensorFlow model

MAX_SEQ_LEN = 100


def save_tf_model_from_transformers():
    model = TFAutoModelForQuestionAnswering.from_pretrained("distilbert-base-cased-distilled-squad")
    callable = tf.function(model.call)
    concrete_function = callable.get_concrete_function([tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="input_ids"),
                                                        tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="attention_mask")])
    model.save('saved_model/distilbert_qa/1', signatures=concrete_function)


def make_predictions():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    context = "My name is Clara and I live in Berkeley."
    question = "What's my name?"
    text = "context:" + context + " question:" + question
    encoded_input = tokenizer(text, pad_to_max_length=MAX_SEQ_LEN, max_length=MAX_SEQ_LEN)
    # TF Serve endpoint
    url = "http://localhost:8501/v1/models/distilbert:predict"

    payload = {"instances": [{"input_ids": encoded_input['input_ids'],
                              "attention_mask": encoded_input['attention_mask']}]
               }
    print(payload)

    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))
    predictions = json.loads(response.text)['predictions']
    print(predictions)


if __name__ == '__main__':
    make_predictions()
