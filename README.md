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
  <img width="1000" alt="2f36546a4514aa8ba744d62432b7b95" src="https://github.com/IntMeGroup/MINT-IQA/assets/104545370/250a0258-1b09-4857-a79d-878f21764726">       


### Database
The constructed AIGCIQA2023 database can be accessed using the links below.
Download AIGCIQA2023 database:[[百度网盘](https://pan.baidu.com/s/1v85j6hKJcRcHm74FDTEosA) 
(提取码：q9dt)], [[Terabox](https://terabox.com/s/1DtV-A9XiuQQDvVPXn6rYvg)]

<img width="500" alt="a82d1d832f524caf4e1b93d4a85eb36" src="https://github.com/IntMeGroup/MINT-IQA/assets/104545370/2a3434b0-2709-4f9e-b514-7559d1cca30d">

The mapping relationship between MOS points and filenames are as follows:

**mosz1**: Quality

**mosz2**: Authenticity

**mosz3**: Correspondence

### Code
<img width="1000" alt="d9cb2afb495449c3ddd397d0cbfe363" src="https://github.com/IntMeGroup/MINT-IQA/assets/104545370/2a53fe69-80e6-4a10-813e-029c8603fe87">


The code of MINT-IQA model will be released to facilitate future research.
### Contact
If you have any question, please contact wangjiarui@sjtu.edu.cn
