from transformers import BertTokenizer, BertModel, BertConfig
import torch

MODEL_NAME = 'bert-base-cased'


def download_pytorch_models():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')\
        .save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))
    bert_model = BertModel.from_pretrained("bert-base-cased").save_pretrained("./{}".format(MODEL_NAME),
                                                                              saved_model=True)
    return bert_tokenizer, bert_model


def get_pytorch_models():
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    bert_tokenizer.save_pretrained('./{}_tokenizer/'.format(MODEL_NAME))
    # bert_model = BertModel.from_pretrained("bert-base-cased", torchscript=True)

    return bert_tokenizer


if __name__ == '__main__':
    encoder = get_pytorch_models()

    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = encoder.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = encoder.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    dummy_input = [tokens_tensor, segments_tensors]

    # Initializing the model with the torchscript flag
    # Flag set to True even though it is not necessary as this model does not have an LM Head.
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072, torchscript=True)

    # Instantiating the model
    model = BertModel(config)

    # The model needs to be in evaluation mode
    model.eval()

    # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
    model = BertModel.from_pretrained(MODEL_NAME, torchscript=True)

    # Creating the trace
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, "./{}".format(MODEL_NAME + ".pt"))
