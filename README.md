# MINT-IQA

This is the official repo of the paper [Understanding and Evaluating Human Preferences
for AI Generated Images with Instruction Tuning](https://arxiv.org/abs/2405.07346):
 ```
@misc{wang2024understanding,
      title={Understanding and Evaluating Human Preferences for AI Generated Images with Instruction Tuning}, 
      author={Jiarui Wang and Huiyu Duan and Guangtao Zhai and Xiongkuo Min},
      year={2024},
      eprint={2405.07346},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
<hr />

> **Abstract:** *Artificial Intelligence Generated Content (AIGC)
has grown rapidly in recent years, among which AI-based image
generation has gained widespread attention due to its efficient
and imaginative image creation ability. However, AI-generated
Images (AIGIs) may not satisfy human preferences due to their
unique distortions, which highlights the necessity to understand
and evaluate human preferences for AIGIs. To this end, in this
paper, we first establish a novel Image Quality Assessment (IQA)
database for AIGIs, termed AIGCIQA2023+, which provides
human visual preference scores and detailed preference explanations from three perspectives including quality, authenticity, and
correspondence. Then, based on the constructed AIGCIQA2023+
database, this paper presents a MINT-IQA model to evaluate and
explain human preferences for AIGIs from Multi-perspectives
with INstruction Tuning. Specifically, the MINT-IQA model first
learn and evaluate human preferences for AI-generated Images
from multi-perspectives, then via the vision-language instruction
tuning strategy, MINT-IQA attains powerful understanding and
explanation ability for human visual preference on AIGIs, which
can be used for feedback to further improve the assessment
capabilities. Extensive experimental results demonstrate that the
proposed MINT-IQA model achieves state-of-the-art performance
in understanding and evaluating human visual preferences for
AIGIs, and the proposed model also achieves competing results
on traditional IQA tasks compared with state-of-the-art IQA
models. The AIGCIQA2023+ database and MINT-IQA model
will be released to facilitate future research.* 
<hr />

![samples_imgs_00](https://github.com/wangjiarui153/AIGCIQA2023/assets/104545370/ab434e91-a766-4de4-babd-1d8fe5cb70c0)
### Database
The constructed AIGCIQA2023 database can be accessed using the links below.
Download AIGCIQA2023 database:[[百度网盘](https://pan.baidu.com/s/1v85j6hKJcRcHm74FDTEosA) 
(提取码：q9dt)], [[Terabox](https://terabox.com/s/1DtV-A9XiuQQDvVPXn6rYvg)]

The mapping relationship between MOS points and filenames are as follows:

**mosz1**: Quality

**mosz2**: Authenticity

**mosz3**: Correspondence
### Code
The code of MINT-IQA model will be released to facilitate future research.
### Contact
If you have any question, please contact wangjiarui@sjtu.edu.cn
