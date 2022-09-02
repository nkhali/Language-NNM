from nltk.parse.generate import generate, demo_grammar
from nltk.corpus import brown
from nltk import CFG
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten
import keras
import tensorflow as tf
import tensorflow.keras.layers
from tensorflow.keras import layers
from tensorflow import keras
import itertools
from tensorflow import keras
from tensorflow.keras import layers
import random

#
# grammar = demo_grammar.split('\n')
# print(grammar)
# grammar.append("hi")
# grammar = "\n".join(grammar)
# print(grammar)
#
# grammar = CFG.fromstring(demo_grammar)
# print(grammar)
#
# for sentence in generate(grammar, n=10):
#     print(' '.join(sentence))
# print("\n")
#
# for sentence in generate(grammar, depth=4):
#     print(' '.join(sentence))
# print("\n")
#
# # len(list(generate(grammar, depth=3)))
# # len(list(generate(grammar, depth=4)))
# # len(list(generate(grammar, depth=5)))
# # len(list(generate(grammar, depth=6)))
# # print(len(list(generate(grammar))))
#
#
# for word in brown.words():
#     print(word)


# non_repeat = []
# for word in brown.words():
#     #print(type(word))
#     if word not in non_repeat:
#         print(word)
#         non_repeat.append(word)
# non_repeat = non_repeat.sort()
# for word in non_repeat:
#     print(word)
# # for genre in brown.categories():
# #     print(genre)

dictionary = ["slapped", "saw", "walked", "slap", "see", "walk", "the", "a", "his", "her", "man", "woman", "park", "dog", "cat", "owl", "friend",
              "in", "with", "on", "to", "above", "could", "would"]
dictionary.sort()
# print(type(dictionary))
# print(dictionary)

# PV = present verb phrase
g = ['', '  S -> DP VP', '  DP -> D NP', '  PP -> P NP', '  AuxP -> Aux PVP', '  S -> DP AuxP',
     "  VP -> 'slapped' DP | 'saw' DP | 'walked' PP",
     "  PVP -> 'slap' DP | 'see' DP | 'walk' PP",
     "  D -> 'the' | 'a' | 'his' | 'her'",
     "  NP -> 'man' | 'woman' | 'park' | 'dog' | 'cat' | 'owl' | 'friend'",
     "  P -> 'in' | 'with' | 'on' | 'to' | 'above'",
     "  Aux -> 'could' | 'would'" '']

# g = ['', '  S -> NP VP', '  NP -> Det N', '  PP -> P NP',
#      "  VP -> 'walked' PP",
#      "  Det -> 'the'",
#      "  N -> 'man'",
#      "  P -> 'in'", '']

# one_hot = np.zeros((len(dict),len(dict)))
# for i in range(len(one_hot)):
#     one_hot[i,i] = 1
# print(one_hot)
a = np.eye(len(dictionary))
print(a)

word_to_hot = dict()
i = 0
for word in dictionary:
    word_to_hot[word] = a[i,:]
    # print(word)
    i+=1
print(word_to_hot)

# Generating sentences for training
g = CFG.fromstring(g)
first_to_second = defaultdict(list)
sentence_list = []
for sentence in generate(g, n=10000):
    sentence_list.append(sentence)
for n in range(0,11):
  print(sentence_list[n])
In = []
Out = []
# k is each sentence and n is each word
for k in sentence_list:
    if sentence_list[0] == k:
        print(k)
    for n in range(len(k)-1):
        # print(n)
        In.append(word_to_hot[k[n]])
        Out.append(word_to_hot[k[n+1]])
# for n in range(0,17):
#   print(In[n])
#   print(Out[n])
print(In[0])
print(Out[0])
In = np.array(In)
Out = np.array(Out)

end = 0
while end == 0:
  for sentence in sentence_list:
    if "could" in sentence and end == 0:
      print(sentence)
      end+=1

# print(In.shape)
# print(Out.shape)
full_text_generated = list(itertools.chain.from_iterable(sentence_list))

inputs = keras.Input(shape=(len(dictionary)))
# Try smaller number of units to see how low we can go to get similar distribution for determiners, aux, etc.
# Trying to get a much smaller hidden representation than the number of words we have
embedding = layers.Dense(6, activation='tanh', name = "embedding")(inputs)

inputs_to_embedding = keras.Model(inputs=inputs, outputs=embedding)
# inputs_to_embedding.compile(
#     loss='BinaryCrossentropy',
#     optimizer='RMSprop',
#     metrics=["accuracy"],
# )
# history1 = inputs_to_embedding.fit(In, Out, epochs=50, batch_size=100, verbose=1)
# # Plotting accuracy
# plt.plot(history1.history['accuracy'])


# model.add(Dense(100, activation='tanh', name = "layer2"))
# this next model.add is the output layer
outputs = layers.Dense(len(dictionary), activation='softmax')(embedding)
model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(
    loss='BinaryCrossentropy',
    optimizer='RMSprop',
    metrics=["accuracy"],
)
history2 = model.fit(In, Out, epochs=50, batch_size=100, verbose=1)

# predictions between inputs and embedding
# !IMPORTANT! If you change the number of words in the dictionary, you have to search for these indices again
predictions = inputs_to_embedding.predict(In)
a_pred_matrix = predictions[31]
the_pred_matrix = predictions[0]
her_pred_matrix = predictions[99]
man_pred_matrix = predictions[1]
woman_pred_matrix = predictions[397]
slapped_pred_matrix = predictions[2]
saw_pred_matrix = predictions[114]
could_pred_matrix = predictions[35134]
would_pred_matrix = predictions[35294]

# Also plot the aux

print('Determiner comparisons:')
print("a:", a_pred_matrix)
print("the:", the_pred_matrix)
print("her:", her_pred_matrix)
print('Noun comparisons:')
print("man:", man_pred_matrix)
print("woman:", woman_pred_matrix)
print('Verb comparisons:')
print("slapped:", slapped_pred_matrix)
print("saw:", saw_pred_matrix)
print('Auxiliary comparisons:')
print("could:", could_pred_matrix)
print("would:", would_pred_matrix)


# # Input word to embedding probability matrix to see if embeddings share similarities among similar word types
# p=0
# print(sentence_list[0:2])
# for word in sentence_list[0]:
#     print(word)
#     print(predictions[p])
#     p+=1
#
# # To get word indices to plug into prediction
# i=0
# for sentence in sentence_list[0:10000]:
#     for word in range(len(sentence)-1):
#         if sentence[word] == "could" or sentence[word] == "would":
#             print(sentence[word], end="(")
#             print(i, end=")\n")
#         i+=1

# Plotting accuracy
plt.plot(history2.history['accuracy'])

# Generate predictions for samples
print(dictionary)
predictions = model.predict(In)
# classes = np.argmax(predictions, axis = 1)
# print(classes[0])
pred_matrix_list = []
prob_matrix_length=0

# # Predicting the next word given a word
# print(sentence_list[0])
for word in range(len(sentence_list[0])):
  print(sentence_list[word])
  print(predictions[word])
  for prob in predictions[word]:
    if prob>0.07:
      # 1d array to list
      list1 = predictions[word].tolist()
      index = list1.index(prob)
      print(dictionary[index])

x = tf.ones((In[0]))
# y = layer2(layer1(x))
print(x)
print(model.layers)

# # Probability matrix for the word 'a'
print(predictions[31])
for prob in predictions[31]:
  if prob>0.07:
    list1 = predictions[31].tolist()
    index = list1.index(prob)
    print(dictionary[index])

# # end = 0
# # while end ==0:
# #   for sentence in sentence_list:
# #     if "a" in sentence and end == 0:
# #       sentence_index = sentence_list.index(sentence)
# #       print(sentence)
# #       print(sentence_index)
# #       end = 1
# #       for word in range(len(sentence_list[sentence_index])):
# #         print(word)
# # a_index = full_text_generated.index('a')
# # print(full_text_generated[a_index])
# # print(predictions[a_index])
# # for word in range(34,40):
# #   print(full_text_generated[word])
# #   print(predictions[word])

# # sentence_index = 0
# # end = 0
# # while end != 1:
# #   for sentence in sentence_list:
# #     if "a" in sentence:
# #       for word in range(len(sentence_list[sentence_index])):
# #         print(predictions[word])
# #         end = 1
# #     else:
# #       sentence_index +=1
# # a_index = sentence_list[sentence_index].index("a")
# # print(a_index)

# Similarity between determiners for input and output: a & the
# Find a way to get index of a in In so I can plug it into predictions[a_index]
# a_pred_matrix = predictions[a_index]
# a_array = a[0]
a_pred_matrix = predictions[31]
the_pred_matrix = predictions[0]
slapped_pred_matrix = predictions[2]
saw_pred_matrix = predictions[114]
man_pred_matrix = predictions[1]
woman_pred_matrix = predictions[397]
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(dictionary, a_pred_matrix, s=10, c='b', marker="x", label='a')
ax1.scatter(dictionary, the_pred_matrix, s=10, c='r', marker="o", label='the')
plt.legend(loc='upper left');
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(dictionary, slapped_pred_matrix, s=10, c='b', marker="x", label='slapped')
ax1.scatter(dictionary, saw_pred_matrix, s=10, c='r', marker="o", label='saw')
plt.legend(loc='upper left');
plt.show()
fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.scatter(dictionary, man_pred_matrix, s=10, c='b', marker="x", label='man')
ax1.scatter(dictionary, woman_pred_matrix, s=10, c='r', marker="o", label='woman')
plt.legend(loc='upper left');
plt.show()

# # example_batch_predictions = model(In)
# # sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=19)
# # sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
# # print("Input:\n", text_from_ids(input_example_batch[0]).numpy())
# # print()
# # print("Next Char Predictions:\n", text_from_ids(sampled_indices).numpy())

print(dictionary)
