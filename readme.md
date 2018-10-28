## <center>MovieReviewSentimentAnalysis（短文本情感分析）</center>

#### 使用TextCNN网络结构: /MovieReviewSentimentAnalysis/src/Model/textCnn.py
#### 使用TextRNN网络结构: /MovieReviewSentimentAnalysis/src/Model/textLstm.py
#### 使用NeuralBagofwords: /MovieReviewSentimentAnalysis/src/Model/NeuralBOW.py
#### 使用FastText网络结构: /MovieReviewSentimentAnalysis/src/Model/FastText.py


### 代码组织结构
<pre><code>
&nbsp;&nbsp;&nbsp;&nbsp;├── data
&nbsp;&nbsp;&nbsp;&nbsp;│   ├── all
&nbsp;&nbsp;&nbsp;&nbsp;│   │   └── all.zip
&nbsp;&nbsp;&nbsp;&nbsp;├── picture
&nbsp;&nbsp;&nbsp;&nbsp;│   └── sentence_length.png
&nbsp;&nbsp;&nbsp;&nbsp;├── readme.md
&nbsp;&nbsp;&nbsp;&nbsp;└── src
&nbsp;&nbsp;&nbsp;&nbsp;    ├── Model
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── FastText.py
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── NeuralBOW.py
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── __init__.py
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── textCnn.py
&nbsp;&nbsp;&nbsp;&nbsp;    │   └── textLstm.py
&nbsp;&nbsp;&nbsp;&nbsp;    ├── count
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── Counter.py
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── __init__.py
&nbsp;&nbsp;&nbsp;&nbsp;    │   └── pd_extract_feature.py
&nbsp;&nbsp;&nbsp;&nbsp;    ├── pkl
&nbsp;&nbsp;&nbsp;&nbsp;    │   ├── test.pkl
&nbsp;&nbsp;&nbsp;&nbsp;    │   └── train.pkl
&nbsp;&nbsp;&nbsp;&nbsp;    ├── preprocess.py
&nbsp;&nbsp;&nbsp;&nbsp;    ├── summary
&nbsp;&nbsp;&nbsp;&nbsp;    ├── test.py
&nbsp;&nbsp;&nbsp;&nbsp;    └── train.py


preprocess.py：数据预处理和特征抽取
train.py：模型训练和评估
