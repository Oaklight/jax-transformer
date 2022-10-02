# Copyright 2020 DeepMind Technologies Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
r"""Trains a transformer for language modeling on a small text dataset.

This example serves to demonstrate:
  - A clean Haiku transformer implementation.
  - An example minimal training loop around it.

This example runs on ASCII text files.
We have not tuned the hyperparameters at all.

Example, using Karpathy's tiny_shakespeare dataset:
$ wget -O /tmp/shakespeare.txt \
    https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
$ python3 examples/transformer/train.py \
    --dataset_path=/tmp/shakespeare.txt --alsologtostderr
"""

import time
from typing import Any, MutableMapping, NamedTuple, Tuple

from absl import app
from absl import flags
from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

import dataset
import model

from datasets import load_dataset
from tqdm.autonotebook import tqdm

from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

import pickle
from datetime import datetime
    
IS_TRAINING = True

LEARNING_RATE = 3e-4
SEQ_LENGTH = 512
GRAD_CLIP_VALUE = 1
LOG_EVERY = 50
MAX_STEPS = 2000
SEED = 42



tokenizer_english = Tokenizer.from_file("vanilla-NMT/en/tokenizer.json")

src_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_english,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="right",
    truncation_side='right',
)

tokenizer_spanish = Tokenizer.from_file("vanilla-NMT/es/tokenizer.json")
tgt_tokenizer = PreTrainedTokenizerFast(
    tokenizer_object=tokenizer_spanish,
    bos_token="<s>",
    eos_token="</s>",
    unk_token="<unk>",
    pad_token="<pad>",
    cls_token="<cls>",
    sep_token="<sep>",
    mask_token="<mask>",
    padding_side="right",
    truncation_side='right',
)

class TrainingState(NamedTuple):
    """Container for the training state."""
    params: hk.Params
    opt_state: optax.OptState
    rng: jnp.DeviceArray
    step: jnp.DeviceArray


def main(_):
    # we use streaming version of dataset
    dataset = load_dataset("avacaondata/europarl_en_es_v2", split='train', streaming=True)

    # encode function to map on each dataset entry
    def encode(examples):
        def decorate(text, tokenizer):
            decorated = f"{tokenizer.bos_token} {text} {tokenizer.eos_token}"
            decorated = decorated.replace('\n', tokenizer.sep_token)
            return decorated

        src_inputs = src_tokenizer(
            decorate(examples['source_en'], src_tokenizer), 
            truncation=True, max_length=SEQ_LENGTH, padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=False,
        )['input_ids']
        tgt_inputs = tgt_tokenizer(
            decorate(examples['target_es'], tgt_tokenizer),
            truncation=True, max_length=SEQ_LENGTH, padding='max_length',
            return_token_type_ids=False,
            return_attention_mask=False,
        )['input_ids']
        return {
            'src_inputs': src_inputs,
            'tgt_inputs': tgt_inputs,
        }

    # now dataset is a iter object
    dataset = iter(dataset.map(encode, remove_columns=["id", "source_en", "target_es", "__index_level_0__"]))
    
    # some training and model parameters:
    CONFIG = model.TransformerConfig(
        input_vocab_size=src_tokenizer.vocab_size,
        output_vocab_size=tgt_tokenizer.vocab_size,
        model_size=256,
        num_heads=8,
        num_layers=6,
        hidden_size=512,
        dropout_rate=0.1,
        src_pad_token=src_tokenizer.pad_token_id,
        tgt_pad_token=tgt_tokenizer.pad_token_id,
    )
    
    # Create the model.
    def forward(
        src_inputs: jnp.ndarray,
        tgt_inputs: jnp.ndarray,
        is_training: bool,
    ) -> jnp.ndarray:

        lm = model.Transformer(
            config=CONFIG,
            is_training=is_training
        )
        return lm(src_inputs, tgt_inputs, is_training=is_training)
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(GRAD_CLIP_VALUE),
        optax.adam(LEARNING_RATE, b1=0.9, b2=0.99),
    )
    
    @hk.transform
    def loss_fn(data) -> jnp.ndarray:
        src_inputs = jnp.asarray(data['src_inputs'], dtype=jnp.int32)[None,:]
        tgt_inputs = jnp.asarray(data['tgt_inputs'], dtype=jnp.int32)[None,:]

        logits = forward(src_inputs, tgt_inputs[:, :-1], IS_TRAINING)
        targets = jax.nn.one_hot(tgt_inputs[:, 1:], CONFIG.output_vocab_size)
        assert logits.shape == targets.shape

        mask = jnp.greater(tgt_inputs[:, :-1], 0)
        log_likelihood = jnp.sum(targets * jax.nn.log_softmax(logits), axis=-1)
        return -jnp.sum(log_likelihood * mask) / jnp.sum(mask) # NLL per token
    
    _Metrics = MutableMapping[str, Any]

    @jax.jit
    def update(state: TrainingState, data) -> Tuple[TrainingState, _Metrics]:
        '''
        Does an SGD step and return metrics
        '''
        rng, new_rng = jax.random.split(state.rng)
        loss_and_grad_fn = jax.is_training(loss_fn.apply)
        loss, gradients = loss_and_grad_fn(state.params, rng, data)

        updates, new_opt_state = optimizer.update(gradients, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)

        new_state = TrainingState(
            params=new_params,
            opt_state=new_opt_state,
            rng=new_rng,
            step=state.step + 1,
        )

        metrics = {
            'step': state.step,
            'loss': loss,
        }
        return new_state, metrics
    
    @jax.jit
    def init(rng: jnp.ndarray, data) -> TrainingState:
        rng, init_rng = jax.random.split(rng)
        initial_params = loss_fn.init(init_rng, data)
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            opt_state=initial_opt_state,
            rng=rng,
            step=np.array(0),
        )
    
    rng = jax.random.PRNGKey(SEED)
    data = next(dataset)
    state = init(rng, data)

    prev_time = time.time()
    for step in range(MAX_STEPS):
        data = next(dataset)
        state, metrics = update(state, data)
        if step % LOG_EVERY == 0:
            step_per_sec = LOG_EVERY / (time.time() - prev_time)
            prev_time = time.time()
            metrics |= {'step_per_sec': step_per_sec}
            logging.info({k: float(v) for k, v in metrics.items()})
    

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%d-%b-%Y (%H:%M:%S)")

    ckpt_file = f'ckpt/state_{timestampStr}.pickle'
    
    # Store data (serialize)
    with open(ckpt_file, 'wb') as handle:
        pickle.dump(state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # # Load data (deserialize)
    # with open(ckpt_file, 'rb') as handle:
    #     unserialized_data = pickle.load(handle)
        
if __name__ == '__main__':
    app.run(main)
