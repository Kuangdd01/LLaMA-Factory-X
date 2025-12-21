import torch
from transformers import AutoTokenizer

from llamafactory.v1.core.trainer_utils.data_collator import DefaultCollator


def test_default_collator():

    def convert_to_hf_messages(messages: list[dict[str, any]]) -> list[dict[str, any]]:
        return [{"role": message["role"], "content": "".join([item["value"] for item in message["content"]]) if isinstance(message["content"], list) else message["content"]} for message in messages]

    # for thinking template check template other place
    processor = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct", trust_remote_code=True, use_fast=True)

    conv1 = [
        {"role": "system", "content": [{"type": "text", "value": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "value": "hi"}]},
        {"role": "assistant", "content": [{"type": "text", "value": "hello"}]},
    ]
    conv2 = [
        {"role": "system", "content": [{"type": "text", "value": "You are a helpful assistant."}]},
        {"role": "user", "content": [{"type": "text", "value": "tell me a joke"}]},
        {"role": "assistant", "content": [{"type": "text", "value": "knock knock"}]},
    ]

    hf_conv1 = convert_to_hf_messages(conv1)
    hf_conv2 = convert_to_hf_messages(conv2)

    hf_conversations = [hf_conv1, hf_conv2]
    texts = processor.apply_chat_template(hf_conversations, tokenize=False, add_generation_prompt=False)
    hf_encoded = processor(texts, add_special_tokens=False, padding=True, return_tensors="pt")

    collator = DefaultCollator(processor=processor)
    batch = collator([conv1, conv2])

    assert torch.equal(hf_encoded["input_ids"], batch["input_ids"])
    assert isinstance(batch["input_ids"], torch.Tensor)
    assert isinstance(batch["labels"], torch.Tensor)
    assert isinstance(batch["attention_mask"], torch.Tensor)
    assert batch["input_ids"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
