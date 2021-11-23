from transformers import BertModel, BertTokenizer
import torch

enc = BertTokenizer.from_pretrained("bert-base-uncased")

if __name__ == '__main__':
    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    dummy_input = [tokens_tensor, segments_tensors]

    # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
    model = BertModel.from_pretrained("bert-base-uncased", torchscript=True)

    # The model needs to be in evaluation mode
    model.eval()

    # Creating the trace
    traced_model = torch.jit.trace(model, dummy_input)
    torch.jit.save(traced_model, "traced_bert.pt")
