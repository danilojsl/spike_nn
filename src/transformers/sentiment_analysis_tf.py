import requests
import numpy as np
import json
import tensorflow as tf
from transformers import AutoTokenizer
from transformers import TFAutoModelForSequenceClassification

MAX_SEQ_LEN = 100


def save_tf_model_from_transformers():
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    callable = tf.function(model.call)
    concrete_function = callable.get_concrete_function([tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="input_ids"),
                                                        tf.TensorSpec([None, MAX_SEQ_LEN], tf.int32, name="attention_mask")])
    model.save('saved_model/distilbert/1', signatures=concrete_function)


def make_predictions():
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    text = "I like you. I love you"
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
    print(softmax(predictions[0]))


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


if __name__ == '__main__':
    save_tf_model_from_transformers()
    # make_predictions()
