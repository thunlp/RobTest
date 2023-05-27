# RobTest

Code and data of the Findings of ACL 2023 paper **"From Adversarial Arms Race to Model-centric Evaluation Motivating a Unified Automatic Robustness Evaluation Framework"**.

## Overview

Textual adversarial attacks can discover models' weaknesses by adding semantic-preserved but misleading perturbations to the inputs. The long-lasting adversarial attack-and-defense arms race in Natural Language Processing (NLP) is algorithm-centric, providing valuable techniques for automatic robustness evaluation. However, the existing practice of robustness evaluation may exhibit issues of incomprehensive evaluation, impractical evaluation protocol, and invalid adversarial samples. In this paper, we aim to set up a unified automatic robustness evaluation framework, shifting towards model-centric evaluation to further exploit the advantages of adversarial attacks. To address the above challenges, we first determine robustness evaluation dimensions based on model capabilities and specify the reasonable algorithm to generate adversarial samples for each dimension. Then we establish the evaluation protocol, including evaluation settings and metrics, under realistic demands. Finally, we use the perturbation degree of adversarial samples to control the sample validity. We implement a toolkit **RobTest** that realizes our automatic robustness evaluation framework. In our experiments, we conduct a robustness evaluation of RoBERTa models to demonstrate the effectiveness of our evaluation framework, and further show the rationality of each component in the framework.

## Dependencies

```
pip install -r requirements.txt
```

Maybe you need to change the version of some libraries depending on your servers.

Then unzip dependency pakage:

```
cd RobTest
unzip naacl2019-like-humans-visual-attacks.zip
```

## Data Preparation

You can use our given dataset (1) agnews (2) jigsaw (2) sst2 in ./data, and you can also use your own dataset.


## Experiments

First, you need to train the victim model with coresponding dataset. You can the example training code.
```
python train.py
```

Then you need to write code for the data and model input interface.

For the models, there is two example:

```python
def load_jigsaw_model():
    evaluated_model = torch.load("jigsaw-roberta-large",map_location=torch.device('cpu'))
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    return evaluated_model, tokenizer

def load_agnews_model():
    evaluated_model = torch.load("ag_newsroberta-large",map_location=torch.device('cpu'))
    tokenizer = AutoTokenizer.from_pretrained('roberta-large')
    return evaluated_model, tokenizer

```

for the datasets, there is an example:
```python
def read_jigsaw(base_path):
    def read_data(file_path):
        data = pd.read_csv(file_path).values.tolist()
        processed_data = []
        for item in data:
            processed_data.append((item[0].strip(),item[1]))  # [sent,label]
        return processed_data   # ([sent,label],[sent,label]...)
    train_path = os.path.join(base_path, 'train.csv')
    test_path = os.path.join(base_path, 'test.csv')
    train, test = read_data(train_path),  read_data(test_path)
    return train, test
```

To conduct the experiments:

```
python robtest.py --mode score --attacker typo --data agnews  --dis_type char --choice both --victim_model textattack/roberta-base-ag-news
```

- model: rule, decision, score, gredient
- degree: range from 0 to 1, default：-1(from 0.1 to 0.6)
- attacker: Typo, Glyph, Phonetic , Synonym, Contextual, Inflection, Syntax, Distraction
- dis_type(attack_type): char(malicious), word(general)
- data: dataset name
- choice: 
- victim_model: name of victim model
- cal_method: average, worst, both
- aug_num: the augmentation number for each sentence


## Citation
Please kindly cite our paper:

```

```

