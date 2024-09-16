
import spacy
import torch
from typing import Optional, List, Iterable, Dict, Any, Tuple

class FineGrainedReward():

    def __init__(self,
                 tokenizer,
                 reward_model_obj,
                 reward_model_att,
                 reward_model_rel,
                 # kl_coef,
                 sep="</s>"
                 ):


        self.reward_model_obj = reward_model_obj
        self.reward_model_att = reward_model_att
        self.reward_model_rel = reward_model_rel

        for param in self.reward_model_obj.parameters():
            param.requires_grad = False
        self.reward_model_obj.eval()

        for param in self.reward_model_att.parameters():
            param.requires_grad = False
        self.reward_model_att.eval()

        for param in self.reward_model_rel.parameters():
            param.requires_grad = False
        self.reward_model_rel.eval()

        self.nlp = spacy.load("en_core_web_sm")

    def get_reward(self, input_ids, labels, attention_mask, images):

        rewards1 = self.reward_model_obj(input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask,
                    images=images)
        rewards2 = self.reward_model_att(input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask,
                    images=images)

        rewards3 = self.reward_model_rel(input_ids=input_ids, labels=labels,
                    attention_mask=attention_mask,
                    images=images)
        return rewards1, rewards2, rewards3


    # def get_reward(self,
    #                prompts_input_ids: torch.tensor,
    #                prompts_attention_mask: torch.tensor,
    #                generated_input_ids: torch.tensor,  # (B, output_len)
    #                generated_attention_mask: torch.tensor,  # (B, output_len)
    #                generated_texts: List[str],
    #                metadata=None,
    #                ):
    #
    #     rewards_output = self.get_finegrained_reward(prompts_input_ids, prompts_attention_mask,
    #                                                  generated_input_ids, generated_attention_mask,
    #                                                  generated_texts, metadata)
    #
    #     return {'rewards/raw': rewards_output['rewards']}


