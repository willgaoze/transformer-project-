#  Sentiment Analysis using BERT

This is A presentation on 2024 spring Transformer  for final project. 

## Table of Contents  

1. [Motivation](#Motivation)  
2. [Overview](#Overview)  
3. [Data](#data)
4. [Model](#model)
5. [Main steps](#Main steps)
6. [Critical Analysis](#critical-analysis)
7. [result](#result)
8. [resources](#resources)

------

## Motivation

In this project, we use Twitter for sentiment analysis because it provides real-time and diverse data reflecting public opinions on current events. We selected the BERT model for its advanced understanding of language and ability to interpret context from both directions, which is crucial for recognizing the subtle meanings in text. BERT's ability to be fine-tuned rapidly and its effective transfer learning make it highly accurate and suitable for complex sentiment classification tasks. These qualities make BERT an excellent choice for developing a sophisticated multi-class sentiment analysis model.

## Overview

This project aims to develop a sentiment analysis model that utilizes the pre-trained BERT model to classify Twitter data into multiple sentiment categories. Employing BERT's advanced linguistic capabilities within the PyTorch framework, this model focuses on multi-class classification to gain deeper insights into public opinions expressed on social media.

**Goals:**

- **Accurate Classification:** The main objective is to precisely categorize tweet sentiments into various classes, offering detailed perspectives on public emotions.
- **Data Preparation and Analysis:** Perform exploratory data analysis, use BERT's tokenizer for data encoding, and establish data loaders for efficient batch processing.
- **Model Optimization:** Enhance the model's performance metrics through structured training cycles, applying optimizers and schedulers to refine BERT for sentiment analysis tasks.
- **Demonstrate Effectiveness:** Show the practical utility of BERT in sentiment analysis by achieving high accuracy and maintaining robust performance in real-world scenarios.



------



## Data

We will use the SMILE Twitter dataset.The SMILE Twitter dataset, collected and annotated for analyzing public emotional responses to cultural experiences in museums, focuses on tweets related to British museums from May 2013 to June 2015. It contains 3,085 tweets categorized into five emotions: anger, disgust, happiness, surprise, and sadness. This dataset is ideal for sentiment analysis research within the specific context of arts and culture, providing a rich source for exploring public sentiment and contributing to better cultural policy-making and museum management practices

![Optional Alt Text](images/c.png)

------

## Model

In our  project ,We utilize the `bert-base-uncased` model from the `transformers` library as our starting point, fine-tuning it on the emotion-labeled tweets. The fine-tuning process adjusts the pre-trained BERT model to better perform sentiment analysis specific to the SMILE dataset.

![Optional Alt Text](images/b.png)

## Main steps

1. **Data Preprocessing**: Clean and format the SMILE dataset, including tokenizing and normalizing text to fit the input requirements of the `bert-base-uncased` model.
2. **Load Tokenizer**: Load tokenizer to encode data.
3. **Model Setting up**: Use a pretrained BERT model. Use AdamW as an optimizer. Use Linear Schedule with Warmup as a scheduler.
4. **Model Fine-tuning**: Using the above setting to finetune the pre-trained model in a training loop.
5. **Model Evaluation**: Evaluate the performance of the model using test dataset with F1 score as rubric.

------

## Critical Analysis

**What needs to be improved for this model?**

The BERT-based sentiment analysis model may struggle with texts containing sarcasm, puns, and non-standard language such as slang and internet speak. These texts often have complex linguistic features and subtle emotional expressions that standard models might not accurately recognize. Additionally, the model may incorrectly interpret minor negative sentiments as neutral or positive due to insufficient or inaccurately labeled training data

**Solution**: To enhance the performance of a BERT-based sentiment analysis model, particularly for complex texts involving sarcasm, double entendres, and non-standard usage, consider the following improvements: Expand and diversify the training dataset to include more examples of complex text types, enhancing the model's generalization capability. Optimize data preprocessing and cleaning to ensure high-quality data, enabling more effective learning. Incorporate advanced architectural features or deeper semantic analysis techniques, such as context-aware semantic parsing, to better recognize subtle differences in text semantics. These strategies can significantly boost the model's efficiency and accuracy in analyzing complex text types.  .

## Result

After fine-tuning, the model achieves a notable accuracy in classifying tweets into their respective emotion categories

![Optional Alt Text](images/a.png)



------



------

## Video Recording

(https://www.youtube.com/watch?v=P3fIrmr6IIE)

------

## Resource

Alammar, J. (2018, December 3). The illustrated bert, elmo, and co. (how nlp cracked transfer learning). The Illustrated BERT, ELMo, and co. (How NLP Cracked Transfer Learning) — Jay Alammar — Visualizing machine learning one concept at a time. Retrieved September 15, 2021, from <https://jalammar.github.io/illustrated-bert/>.

Anastassiou, A. (n.d.). Sentiment Analysis with Deep Learning using BERT (MOOC). Coursera. <https://www.coursera.org/projects/sentiment-analysis-bert>.

Chaffey, D. (2021, July 23). Global social media statistics research summary [updated 2021]. Smart Insights. Retrieved September 16, 2021, from <https://www.smartinsights.com/social-media-marketing/social-media-strategy/new-global-social-media-research/>.

Devlin, J., Chang, M., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. NAACL. <https://arxiv.org/abs/1810.04805>.

Peters, M. E., Neumann, M., Iyyer, M., Gardner, M., Clark, C., Lee, K. & Zettlemoyer, L. (2018). Deep contextualized word representations (cite arxiv:1802.05365Comment: NAACL 2018.

Taylor, W. L. (1953). “Cloze procedure”: a new tool for measuring readability. Journalism Quarterly, 30, 415–433. <https://www.gwern.net/docs/psychology/writing/1953-taylor.pdf>.



