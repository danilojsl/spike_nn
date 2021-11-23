"""""
Torchscript with HuggingFace
https://huggingface.co/transformers/torchscript.html
"""""

from transformers import BertModel, BertTokenizer
import torch

enc = BertTokenizer.from_pretrained("bert-base-cased")
TS_MODEL_PATH = "../../models/traced_bert_cased.pt"


# The dummy inputs are used to do a model forward pass.
def get_dummy_input(text):
    # Tokenizing input text
    tokenized_text = enc.tokenize(text)

    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = '[MASK]'
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]

    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    return [tokens_tensor, segments_tensors]


def save_torchscript_model(dummy_input):
    # If you are instantiating the model with `from_pretrained` you can also easily set the TorchScript flag
    model = BertModel.from_pretrained("bert-base-cased", torchscript=True)

    # The model needs to be in evaluation mode
    model.eval()

    # Creating the trace
    dummy_input_tensor = torch.concat(dummy_input)
    traced_model = torch.jit.trace(model, dummy_input_tensor)
    torch.jit.save(traced_model, TS_MODEL_PATH)


def load_torchscript_model(dummy_input):
    loaded_model = torch.jit.load(TS_MODEL_PATH)
    loaded_model.eval()

    dummy_input_tensor = torch.concat(dummy_input)  # dtype = torch.int64
    all_encoder_layers, pooled_output = loaded_model(dummy_input_tensor)
    print(all_encoder_layers)  # dtype = tprch.float32
# org.scalatest.flatspec.AnyFlatSpec


if __name__ == '__main__':
    sentence1 = "Who was Jim Henson ?"
    sentence2 = "Jim Henson was a puppeteer"
    long_sentence1 = "A touch of red over the lips of Esther had strayed beyond their outline; the yellow on her" \
                     " dress was spread with such unctuous plumpness as to have acquired a kind of solidity, and stood" \
                     " boldly out from the receding atmosphere"
    long_sentence2 = "All these things and, still more than these, the treasures which had come to the church from " \
                     "personages who to me were almost legendary figures, because of which I used to go forward into" \
                     " the church when we were making our way to our chairs as into a fairy-haunted valley, where the" \
                     " rustic sees with amazement on a rock, a tree, a marsh, the tangible proofs of the little " \
                     "people's supernatural passage—all these things made of the church for me something entirely" \
                     " different from the rest of the town; a building which occupied, so to speak, four dimensions of" \
                     " space—the name of the fourth being Time—which had sailed the centuries with that old nave," \
                     " where bay after bay, chapel after chapel, seemed to stretch across and hold down and conquer " \
                     "not merely a few yards of soil, but each successive epoch from which the whole building had " \
                     "emerged triumphant, hiding the rugged barbarities of the eleventh century in the thickness of" \
                     " its walls, through which nothing could be seen of the heavy arches, long stopped and blinded" \
                     " with coarse blocks, except where, near the porch, a deep groove was furrowed into one" \
                     " wall by the tower-stair; and even there the barbarity was veiled by the graceful gothic arcade" \
                     " which pressed coquettishly upon it, like a row of grown-up sisters who, to hide him from the" \
                     " eyes of strangers, arrange themselves smilingly in front of a countrified, unmannerly and" \
                     " ill-dressed younger brother; rearing into the sky above the Square a tower which had looked" \
                     " down upon Louis, and seemed to behold him still; and thrusting down with its crypt into the" \
                     " blackness of a Merovingian night, through which, guiding us with groping finger-tips beneath " \
                     "the shadowy vault, ribbed strongly as an immense bat's wing of stone, Théodore would light up " \
                     "for us with a candle the tomb of Sigebert's little daughter, in which a deep hole, like the bed" \
                     " of a fossil, had been bored, or so it was said, by a crystal lamp which, on the night when the" \
                     " Frankish princess was murdered, had left, of its own accord, the golden chains by which it was" \
                     " suspended where the apse is to-day and with neither the crystal broken nor the light " \
                     "extinguished had buried itself in the stone, through which it had gently forced its way"
    sentence3 = "And then the apse of Combray: what am I to say of that?"
    my_text = f"[CLS] {long_sentence2} [SEP] {sentence3} [SEP]"
    my_dummy_input = get_dummy_input(my_text)

    save_torchscript_model(my_dummy_input)

    # load_torchscript_model(my_dummy_input)
