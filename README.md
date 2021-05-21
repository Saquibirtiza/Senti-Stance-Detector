# Senti-Stance-Detector
## Overview

In this project, we developed a stance detection system that uses a separate affect processing system to improve the accuracy of its predicted stance labels. The affect of choice in this case was the sentiment of the tweets which tells whether the user has a positive or a negative sentiment when writing the tweet. To give an overview of our approach, we first generate sentiment labels using a dedicated sentiment classification model and then use those labels as an additional feature in the stance detection model. The flow diagram in Figure 1 gives a visual illustration of how our model works.

![Picture1](https://user-images.githubusercontent.com/30957097/119084528-9424c400-b9c7-11eb-8251-71e6afac2420.png)

It is evident from the diagram that our approach is a simple three step process at the end of which we obtain stance labels for the tweets. Initially, we preprocess the tweets to:
-	Remove any web links from the tweet if available.
-	Remove retweets and user information if available.
-	Remove punctuation, stopwords and numbers.
-	Fix any inconsistent spacing between words.
-	Convert all words to lowercase.
-	Apply word rooter for stemming.

Preprocessing is a crucial step in any natural language processing task and can greatly affect the performance of the model.  That is why we ensured that the data is preprocessed properly. Following this step, a sentiment classification model is trained on another COVID twitter dataset containing about 5000 training instances. This model helps to assign sentiment labels to the Twitter data preprocessed in the previous step. Finally, in the last step, we use the modified dataset to train a stance detector. As the diagram shows, each instance is labeled one of the four classes it is most probable to belong to. The following sections will elaborate on the individual classifiers and the architectures they use.

## Sentiment detector
### The architecture, input, and output
As Figure 2 shows, we use an LSTM based model for our sentiment recognition system. It takes as input tokenized sentences where each word makes up one node in the embedding layer. These embeddings are then used in Bi-LSTM layer to generate hidden values which are in turn passed through two fully connected layers and a Softmax layer before it is finally categorized into one of the two classes: positive or negative. 

![Picture2](https://user-images.githubusercontent.com/30957097/119084613-bcacbe00-b9c7-11eb-8afa-13ee4d96a82d.png)

The detailed architecture along with the dimension of the input and output values of each layer is given below:
```
LSTM_net(
  (embedding): Embedding(6640, 100, padding_idx=1)
  (rnn): LSTM(100, 256, num_layers=2, dropout=0.2, bidirectional=True)
  (fc1): Linear(in_features=512, out_features=256, bias=True)
  (fc2): Linear(in_features=256, out_features=1, bias=True)
  (dropout): Dropout(p=0.2, inplace=False)
)
```

### How was sentiment recognition tuned?
To ensure that the sentiment classification model is well-tuned for our training dataset, we use a COVID tweet dataset of around 5000 instances to initially train our model as mentioned earlier. We choose this dataset particularly because the contents of the tweets in this dataset are similar to the dataset for which we need to find the labels. So, training the model on this dataset ensures that it learns features that are similar to our training dataset and are sufficiently capable to classify new instances into one of the two classes. Our model achieves a validation accuracy of around 55% after 50 epochs the plots of which are given as follows: 

![Picture3](https://user-images.githubusercontent.com/30957097/119084993-6ee48580-b9c8-11eb-900d-bd6fb896d613.png)

The equations used in each layer of the model are given as follows:

Embedding Layer:
- x=(E_1,E_2,E_3,E_4…E_n)

Bi-LSTM layer:
- i_t= σ(U_i h_(t-1)+ W_i x_i)
- f_t= σ(U_f h_(t-1)+ W_f x_t)
- o_t= σ(U_o h_(t-1)+ W_o x_t)
- h_t= o_t tan h⁡(c_t )
- k_t= c_(t-1)  .f_t
- j_t= g_t  .i_t
- g_t= tanh(U_g h_(t-1)+ W_g x_t)
- c_t= j_t+k_t

Fully Connected layer:
- m=Fh
- z=Sm

Dropout layer:
- y=Softmax(z)

## Stance detector
### Overview:
For our stance classification model, we use the tweet and tokenize sentiment labels as input for the model. The sentiment labels are assigned a label of 0 for negative and 1 for positive sentiment. The misinformation target included in the dataset is not used during the training phase. Instead, separate models are trained for each misinformation target. To do so, the dataset is initially split into subsets according to their misinformation id and then these subsets of data are then used to train individual models as shown in Figure 4. During the testing phase, the appropriate model based on the misinformation target of that instance is chosen and used to classify the instance. The overall accuracy of the model depends on how many of the instances in the development set are correctly labeled by all the models combined.
The architecture, input, and output:

Figure 5 shows the detailed architecture that we use for our stance detection model. It has two separate segments, one using the tweet text data and the other using the sentiment labels of the tweets as input. The outputs of these segments are concatenated into one single layer and further transformations are performed on it. The final output that we have is a stance label that is the most fitting for the tweet. 
For the tweet text data, we pass it through an embedding layer first and then through a bidirectional LSTM layer to capture the sequential information of the text data. Since the sentiment labels are categorical, we convert them into numerical tokens as explained earlier, and pass them through two consecutive dense layers to extract high-level features for the input.  These high-level features are then concatenated with the output of the LSTM layer so that both the features could be used to detect the stance labels.  

After further transformations on this concatenated layer, we obtain a probability distribution showing which class the tweet is most likely to belong to. The class with the highest probability is assigned as the label of the tweet. Figures 6 and 7 show the input and output dimension of each layer and how the data flows through the model. 

### Result:
For a development set containing 25% of the total data, we obtain an accuracy of around 43.7% when we use both the tweet and the sentiment labels as input which is slightly higher than what we obtain when we only use tweet text data as input. This shows that the additional feature does help in improving the classification of the stance.

![Picture4](https://user-images.githubusercontent.com/30957097/119084650-cc2c0700-b9c7-11eb-9f22-ee593fb11250.png)

![Picture5](https://user-images.githubusercontent.com/30957097/119084669-d5b56f00-b9c7-11eb-88c8-98ead4c59d9c.png)

![Picture6](https://user-images.githubusercontent.com/30957097/119084694-de0daa00-b9c7-11eb-9401-6e1e015eefb4.png)

![Picture7](https://user-images.githubusercontent.com/30957097/119084707-e5cd4e80-b9c7-11eb-9c01-5f9f7e726000.png)

### Collaboration
I worked with student X1 to initially combine all the Twitter data files, convert them from jsonl to text, and then split them into a train and development set using a ratio that gives the optimal result. After several experiments, we settled on using a 75:25 ratio for the split since it yielded the highest accuracy on the base stance detection model. We also ensured that for all our experiments we used the same training and testing data so that we can compare the performance of our different models later. 
I specifically tried improving the results of his BERT model by including sentiment features generated by my sentiment detector by appending them to the end of the tweets. I also extended on his base LSTM stance detection model and concatenated to it high-level features from another LSTM model that was trained on my predicted sentiment labels. The model that performed the best for both the tweet and the sentiment data was chosen and elaborated in this article.


