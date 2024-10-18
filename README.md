# FGAIF
1. train the reward model with the script train_fine_reward_model.sh.
2. train the policy model with  the script train_fgrl_model.sh.


If you find this repo useful for your research, please consider citing the paper

FGAIF:
```bibtex
@misc{jing2024fgaifaligninglargevisionlanguage,
      title={FGAIF: Aligning Large Vision-Language Models with Fine-grained AI Feedback}, 
      author={Liqiang Jing and Xinya Du},
      year={2024},
      eprint={2404.05046},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2404.05046}, 
}
```

FaiithScore:

```bibtex
@misc{jing2024faithscorefinegrainedevaluationshallucinations,
      title={FaithScore: Fine-grained Evaluations of Hallucinations in Large Vision-Language Models}, 
      author={Liqiang Jing and Ruosen Li and Yunmo Chen and Xinya Du},
      year={2024},
      eprint={2311.01477},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2311.01477}, 
}
```

LLaVA-RLHF:

```bibtex
@article{sun2023aligning,
  title={Aligning large multimodal models with factually augmented rlhf},
  author={Sun, Zhiqing and Shen, Sheng and Cao, Shengcao and Liu, Haotian and Li, Chunyuan and Shen, Yikang and Gan, Chuang and Gui, Liang-Yan and Wang, Yu-Xiong and Yang, Yiming and others},
  journal={arXiv preprint arXiv:2309.14525},
  year={2023}
}
```

LLaVA:

```bibtex
@misc{liu2023llava,
      title={Visual Instruction Tuning},
      author={Liu, Haotian and Li, Chunyuan and Wu, Qingyang and Lee, Yong Jae},
      publisher={arXiv:2304.08485},
      year={2023},
}
```

## Acknowledgements

We thank [Meta LLaMA team](https://github.com/facebookresearch/llama), [Standford Alpaca team](https://github.com/tatsu-lab/stanford_alpaca), [Vicuna team](https://github.com/lm-sys/FastChat), [LLaVA team](https://github.com/haotian-liu/LLaVA), [QLoRA team](https://github.com/artidoro/qlora), [Hugging Face PEFT](https://github.com/huggingface/peft), [LLaVA-RLHF team](https://github.com/llava-rlhf/LLaVA-RLHF), and [AlpacaFarm team](https://github.com/tatsu-lab/alpaca_farm) for their open-source efforts in democratizing large language models.
