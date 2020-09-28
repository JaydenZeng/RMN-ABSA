# RMN
tensorflow implementation for RMN
## Requirements
   spacy==2.0.18 <br>
   tensorflow-gpu==1.15.0 <br>
   numpy==1.15.4 <br>
   sklearn <br>
   python 3.6.9 <br>
   
## Usage
* install requirements <br>
```python 
pip install -r requirements.txt
```
  
* download en-core-web-sm <br>
```python 
python -m spacy download en
```
   
* download 'glove.840B.300d.zip' from [here](https://github.com/stanfordnlp/GloVe#download-pre-trained-word-vectors) and extract 'glove.840B.300d.txt' to './data' <br>
* generate word2idx.pkl and glove_300.pkl matrix <br>
```python 
python preprocess.py
```
* generate train and test.pkl
```python 
python Dataset.py
```
* train (can modify settings in Settings.py) <br>
```python 
python train.py
``` 
* test <br>
```python 
python train.py --train=False
```
* Tips <br>
for finetune.py, you can get the current best model, and finetune it with smaller learning rate or remove dropout to train new model.
