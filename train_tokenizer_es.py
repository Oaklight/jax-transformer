import os

from datasets import load_dataset
from tqdm.autonotebook import tqdm

dataset = load_dataset("avacaondata/europarl_en_es_v2", split='train')
# dataset = load_dataset("opus_books", "en-es", split='train')
def get_source_corpus():
    for i in range(0, len(dataset), 1000):
        yield dataset[i : i + 1000]["target_es"]
        
from tokenizers import (Regex, Tokenizer, decoders, models, normalizers,
                        pre_tokenizers, processors, trainers)

tokenizer = Tokenizer(models.Unigram())

tokenizer.normalizer = normalizers.Sequence(
    [
        normalizers.Replace("``", '"'),
        normalizers.Replace("''", '"'),
        normalizers.NFKD(),
        normalizers.StripAccents(),
        normalizers.Replace(Regex(" {2,}"), " "),
    ]
)

tokenizer.pre_tokenizer = pre_tokenizers.Metaspace()
tokenizer.pre_tokenizer.pre_tokenize_str("Let's test the pre-tokenizer!")

special_tokens = ["<cls>", "<sep>", "<unk>", "<pad>", "<mask>", "<s>", "</s>"]
trainer = trainers.UnigramTrainer(
    vocab_size=25000, special_tokens=special_tokens, unk_token="<unk>"
)

tokenizer.train_from_iterator(get_source_corpus(), trainer=trainer)
os.makedirs('vanilla-NMT/es', exist_ok=True)
tokenizer.save('vanilla-NMT/es/tokenizer.json')
