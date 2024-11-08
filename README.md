<h1 align="center">
  <a href="https://baleegh-production.up.railway.app/">جرب بليغ</a>
</h1>

# Folder Structure:

## Baleegh: The base code for training and data collection
- train: All training code, mainly on modal.com serverless training.
- data:
  1. Raw: The base data sources and each of them cleaned based on its own source and category.
  2. Processed: All data sources combined to 1 dataset and general cleaning performed on them. Then uploaded to [Huggingface Repo](https://huggingface.co/datasets/Abdulmohsena/Classic-Arabic-English-Language-Pairs)
- eval: Creation and management of multiple evaluation metrics, and the testing of multiple models to choose the best version.
- model preparation: Many ways we tried to compress, enhance, or prepare the base model for better training outcomes.
- Inference Strategy.ipynb: The followed inference strategy to output the best result. (May be outdated)

## Frontend: The frontend code for the project

## Backend: the backend code for hosting the model and performing the inference strategy.

## Preview
![image](https://github.com/user-attachments/assets/193fb3d0-a2e9-465d-8d41-4ccf7e732dcc)
