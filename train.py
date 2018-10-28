from collections import Counter
import numpy as np

import time
import sys
import numpy as np

import argparse


positive_counts = Counter()
negative_counts = Counter()
total_counts = Counter()

g = open('reviews.txt','r')
reviews = list(map(lambda x:x[:-1],g.readlines()))
g.close()

g = open('labels.txt','r')
labels = list(map(lambda x:x[:-1].upper(),g.readlines()))
g.close()

for i in range(len(reviews)):
    for word in reviews[i].split(' '):
        if(labels[i]=='POSITIVE'):
            positive_counts.update([word])
            total_counts.update([word])
        elif(labels[i]=='NEGATIVE'):
            negative_counts.update([word])
            total_counts.update([word])


pos_neg_ratios = Counter()


for t, c in list(total_counts.most_common()):
    if(c>=100):
        pos_neg_ratios.update({t: positive_counts[t]/float(negative_counts[t]+1)})


for t, c in list(pos_neg_ratios.most_common()):
    pos_neg_ratios[t]= np.log(c)



vocab= set(total_counts.keys())
vocab_size = len(vocab)

layer_0 = np.zeros(shape=(1,vocab_size))


word2index = {}
for i,word in enumerate(vocab):
    word2index[word] = i

def update_input_layer(review):

    global layer_0
    layer_0 *= 0


    for words in review.split(' '):
        layer_0[0][word2index[words]]+=1

def get_target_for_label(label):
    

    return 1 if label=="POSITIVE" else 0



class SentimentNetwork:
    def __init__(self, reviews,labels, min_count = 10, polarity_cutoff = 0.1, hidden_nodes = 10, learning_rate = 0.1):


        np.random.seed(1)

        self.pre_process_data(reviews, labels, polarity_cutoff, min_count)

        self.init_network(len(self.review_vocab),hidden_nodes, 1, learning_rate)

    def pre_process_data(self, reviews, labels, polarity_cutoff, min_count):

        positive_counts= Counter()
        negative_counts= Counter()
        total_counts= Counter()

        for i in range(len(reviews)):
            if labels[i]=="POSITIVE":
                for word in reviews[i].split(" "):
                    positive_counts[word]+=1
                    total_counts[word]+=1
                else:
                    negative_counts[word]+=1
                    total_counts[word]+=1

        pos_neg_ratios= Counter()

        for term, cnt in list(total_counts.most_common()):
            if(cnt>=50):
                pos_neg_ratio= positive_counts[term]/float(negative_counts[term]+1)
                pos_neg_ratios[term]= pos_neg_ratio

        for word, ratio in pos_neg_ratios.most_common():
            if(ratio>1):
                pos_neg_ratios[word]= np.log(ratio)
            else:
                pos_neg_ratios[word]= -np.log(1/(ratio+0.01))


        review_vocab = set()
        for review in reviews:
            for word in review.split(" "):
                if(total_counts[word]>min_count):

                    if word in pos_neg_ratios.keys():
                        if((pos_neg_ratios[word]>=polarity_cutoff) or (pos_neg_ratios[word]<=-polarity_cutoff)):
                            review_vocab.add(word)
                    else:
                        review_vocab.add(word)


        self.review_vocab = list(review_vocab)


        label_vocab = set()
        for label in labels:
            label_vocab.add(label)


        self.label_vocab = list(label_vocab)


        self.review_vocab_size = len(self.review_vocab)
        self.label_vocab_size = len(self.label_vocab)


        self.word2index = {}
        for i, word in enumerate(self.review_vocab):
            self.word2index[word] = i


        self.label2index = {}
        for i, label in enumerate(self.label_vocab):
            self.label2index[label] = i

    def init_network(self, input_nodes, hidden_nodes, output_nodes, learning_rate):

        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes

        self.learning_rate = learning_rate

        self.weights_0_1 = np.zeros((self.input_nodes,self.hidden_nodes))


        self.weights_1_2 = np.random.normal(0.0, self.output_nodes**-0.5, (self.hidden_nodes, self.output_nodes))


        self.layer_1 = np.zeros((1,hidden_nodes))


    def get_target_for_label(self,label):
        if(label == 'POSITIVE'):
            return 1
        else:
            return 0

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_output_2_derivative(self,output):
        return output * (1 - output)

    def train(self, training_reviews_raw, training_labels):

        training_reviews = list()
        for review in training_reviews_raw:
            indices = set()
            for word in review.split(" "):
                if(word in self.word2index.keys()):
                    indices.add(self.word2index[word])
            training_reviews.append(list(indices))

        assert(len(training_reviews) == len(training_labels))


        correct_so_far = 0


        start = time.time()


        for i in range(len(training_reviews)):

            review = training_reviews[i]
            label = training_labels[i]


            self.layer_1 *= 0
            for index in review:
                self.layer_1 += self.weights_0_1[index]


            layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))

            layer_2_error = layer_2 - self.get_target_for_label(label)
            layer_2_delta = layer_2_error * self.sigmoid_output_2_derivative(layer_2)


            layer_1_error = layer_2_delta.dot(self.weights_1_2.T)
            layer_1_delta = layer_1_error


            self.weights_1_2 -= self.layer_1.T.dot(layer_2_delta) * self.learning_rate


            for index in review:
                self.weights_0_1[index] -= layer_1_delta[0] * self.learning_rate


            if(layer_2 >= 0.5 and label == 'POSITIVE'):
                correct_so_far += 1
            elif(layer_2 < 0.5 and label == 'NEGATIVE'):
                correct_so_far += 1

            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

    def test(self, testing_reviews, testing_labels):

        correct = 0

        start = time.time()

        for i in range(len(testing_reviews)):
            pred = self.run(testing_reviews[i])
            if(pred == testing_labels[i]):
                correct += 1


            elapsed_time = float(time.time() - start)
            reviews_per_second = i / elapsed_time if elapsed_time > 0 else 0

    def run(self, review):

        self.layer_1 *= 0
        unique_indices = set()
        for word in review.lower().split(" "):
            if word in self.word2index.keys():
                unique_indices.add(self.word2index[word])
        for index in unique_indices:
            self.layer_1 += self.weights_0_1[index]


        layer_2 = self.sigmoid(self.layer_1.dot(self.weights_1_2))


        if(layer_2[0] >= 0.8):
            return " * * * * *"
        elif(layer_2[0] >= 0.6):
            return " * * * *"
        elif(layer_2[0] >= 0.4):
            return " * * *"
        elif (layer_2[0] >= 0.2):
            return " * *"
        else:
            return " *"

mlp = SentimentNetwork(reviews[:-1000],labels[:-1000],min_count=20,polarity_cutoff=0.8,learning_rate=0.01)
mlp.train(reviews[:-1000],labels[:-1000])


parse= argparse.ArgumentParser()
parse.add_argument('--rev', type=str, default='Best Movie Ever')
arg= parse.parse_args()
print(' ')
print('Review: ', arg.rev)
print('Stars:  ', mlp.run(arg.rev))
