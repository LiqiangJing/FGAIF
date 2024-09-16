# Copyright 2023 The LLaVA-RLHF Team
# Copyright 2023 The Alpaca Team
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

import argparse
import glob
import os
import random
from typing import (
    Callable,
    Dict,
    Optional,
    Sequence,
    Union,
    Mapping,
    Any,
)

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import transformers
from llava import conversation as conversation_lib
from llava.constants import (
    IGNORE_INDEX,
    IMAGE_TOKEN_INDEX,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IM_END_TOKEN,
)
from llava.train.train import DataArguments

from data_utils.constants import FACTUAL_PROMPT

def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    # prompt = "hello<image>good "
    prompt = "<image>\n" + prompt
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]
    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    # [[-200, -200], [-200, -200]] IMAGE_TOKEN_INDEX
    # [[1, 22172], [-200, -200], [1, 1781, 29871]]
    # [1, 22172, -200, 1781, 29871]  1 and 29871 are special tokens

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    mask_target: bool = True,
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence["value"]:
                sentence["value"] = (
                    sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
                )
                sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
                sentence["value"] = sentence["value"].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence["value"] = sentence["value"].replace(
                        DEFAULT_IMAGE_TOKEN,
                        "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>",
                    )
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = (
                    DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
                )
            sentence["value"] = sentence["value"].replace(
                DEFAULT_IMAGE_TOKEN, replace_token
            )

    return sources


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    mask_target: bool = True,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    reward_model_prompt: Optional[str] = None,
    image_captions: Optional[Sequence[str]] = None,
    fine_labels: Optional[int] = None,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    # roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
    # print(conv.get_prompt()) A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
    # roles = {'human': 'USER', 'gpt': 'ASSISTANT'}
    # [[{'from': 'human', 'value': '<image>\nWhat colors are the boats in the image?'},
    #   {'from': 'gpt', 'value': 'The boats in the image are blue, white, and black.'},
    #   {'from': 'human', 'value': 'What is the general setting of the image?'}, {'from': 'gpt',
    #                                                                             'value': 'The image depicts a harbor with boats tied up near the land, and several cranes can be seen in the background, indicating an industrial or commercial waterfront area.'},
    #   {'from': 'human', 'value': 'What are the boats preparing to do?'},
    #   {'from': 'gpt', 'value': 'The boats are lined up, likely preparing for a day of fishing out on the water.'},
    #   {'from': 'human', 'value': 'What type of body of water are the boats on?'}, {'from': 'gpt',
    #                                                                                'value': 'The boats are on a large body of water, possibly an ocean, considering their proximity to the land.'},
    #   {'from': 'human', 'value': 'Can any other structures be identified in the image besides the cranes?'},
    #   {'from': 'gpt',
    #    'value': 'Yes, there are a few buildings visible along the shore, which contribute to the industrial or commercial appearance of the waterfront area.'}]]

    # Apply prompt templates
    # has_image True
    conversations = []
    # print(len(sources))  result: 1
    text = "Prompt: Describe this image in detail. \n"

    for i, source in enumerate(sources):
        # if image_captions is not None:
        #     text += "Captions: "
        #     for caption in image_captions[i]:
        #         text = text + f"  - {caption}\n"
        for sub_sen in source:
            text += sub_sen + "</s>"
        conversations.append(text)

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # print(conversations)conversations ## ["sssss ss]
    # print(input_ids.shape)
    # print(input_ids[0][:10]) # [1, -200, ...]
    # iddd = input_ids[0].tolist()
    # print(input_ids)
    # print(tokenizer.decode([iddd[0]]+iddd[2:], skip_special_tokens=False))
    # print(tokenizer.convert_tokens_to_ids(['</s>']))
    targets = input_ids.clone()
    ## mask 2 is the id of </s>
    targets[targets != 2] = IGNORE_INDEX
    validity = [True] * len(input_ids)
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    return dict(
        input_ids=input_ids,
        labels=targets,
        validity=validity,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    mask_target: bool = True,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    reward_model_prompt: Optional[str] = None,
    image_captions: Optional[Sequence[str]] = None,
    fine_labels: torch.Tensor = None,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """

    # TODO(sheng): hack for LLAVA_V1, LLAMA_2
    if (
        conversation_lib.default_conversation.sep_style
        == conversation_lib.SeparatorStyle.LLAMA_2
    ):
        # print("utilize preprocess llama 2------------")
        return preprocess_llama_2(
            sources,
            tokenizer,
            has_image=has_image,
            mask_target=mask_target,
            query_len=query_len,
            response_len=response_len,
            reward_model_prompt=reward_model_prompt,
            image_captions=image_captions,
        )
    if conversation_lib.default_conversation.version.startswith("v1"):
        # print("utilize preprocess v1------------")
        # print(image_captions)
        # print(has_image)
        ### run with this
        return preprocess_v1(
            sources,
            tokenizer,
            has_image=has_image,
            mask_target=mask_target,
            query_len=query_len,
            response_len=response_len,
            reward_model_prompt=reward_model_prompt,
            fine_labels=fine_labels,
            image_captions=image_captions,
        )

    raise NotImplementedError



def preprocess4reward(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = True,
    image_captions: Optional[Sequence[str]] = None,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()

    # has_image True
    conversations = []
    # print(len(sources))  result: 1
    text = "Prompt: Describe this image in detail. \n"

    for i, source in enumerate(sources):
        if image_captions is not None:
            text += "Captions: "
            for caption in image_captions[i]:
                text = text + f"  - {caption}\n"
        for sub_sen in source:
            text += sub_sen + "</s>"
        conversations.append(text)

    if has_image:
        input_ids = torch.stack(
            [
                tokenizer_image_token(prompt, tokenizer, return_tensors="pt")
                for prompt in conversations
            ],
            dim=0,
        )
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # print(conversations)conversations ## ["sssss ss]
    # print(input_ids.shape)
    # print(input_ids[0][:10]) # [1, -200, ...]
    # iddd = input_ids[0].tolist()
    # print(input_ids)
    # print(tokenizer.decode([iddd[0]]+iddd[2:], skip_special_tokens=False))
    # print(tokenizer.convert_tokens_to_ids(['</s>']))
    targets = input_ids.clone()
    ## mask 2 is the id of </s>
    targets[targets != 2] = IGNORE_INDEX
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO
    return dict(
        input_ids=input_ids,
        labels=targets,
    )
