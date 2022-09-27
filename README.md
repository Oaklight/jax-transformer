# Resources about transformer in jax:
- Attention is all you need: https://arxiv.org/abs/1706.03762
- Annotated transformer (Pytorch): https://nlp.seas.harvard.edu/2018/04/03/attention.html
- einsum:
    - videos:
        - Youtube: https://youtu.be/ULY6pncbRY8
        - Bilibili: https://www.bilibili.com/video/BV1ee411g7Sv?share_source=copy_web&vd_source=4b995a4a830cf8658351db7c6e2b0d08
    - code snippets: https://numpy.org/doc/stable/reference/generated/numpy.einsum.html
    - einops: https://github.com/arogozhnikov/einops/
- Jax:
    - Jax: https://jax.readthedocs.io/en/latest/index.html
    - Haiku: https://dm-haiku.readthedocs.io/en/latest/index.html
- reference implementations I used yesterday:
    - haiku:
        this is cleaner but more functional & the transformer is not complete
        - https://github.com/deepmind/dm-haiku/blob/c18be3df5e85796492f2915af261b5517f12bacc/examples/transformer/model.py
        - https://github.com/deepmind/dm-haiku/blob/c18be3df5e85796492f2915af261b5517f12bacc/haiku/_src/attention.py        
    - flax:
        this is more complex and easier to get lost
        - https://github.com/google/flax/blob/6dba29098fba23a457e87f104bfef2704dbf54cd/examples/wmt/models.py
        - https://github.com/google/flax/blob/cc88a73f5cf3d5970981c104364bc5864841db1a/flax/linen/attention.py
    - elegy:
        perhaps use it for training loop
        - https://github.com/poets-ai/elegy