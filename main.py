import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB

def process_text(D):
  processed = []

  for text in D:
    lower = text.toLowerCase()
    split = lower.split(' ')

class NaiveBayes:

  def __init__(self):
    self.condprob = {}
    self.prior = {}
    self.classes = []
    self.xprob = {}

  
  def fit(self, X, y):
    self.classes = np.unique(y)
    
    for c in self.classes:
      examples = X[y == c]
      
      self.prior[c] = len(examples) / len(y)

      for i in range(len(X[0])):
        if c == self.classes[0]:
          self.condprob[i] = {}
        self.condprob[i][c] = (1 + np.sum(examples[:,i])) / (len(examples) + 1)
      
    
      
  def score(self, X, y):
    correct = 0
    for i in range(len(X)):
      prediction = self.predict([X[i]])

      if prediction == y[i]:
        correct += 1
    
    return correct / len(X)

  def predict(self, X):
    results = []
    for x in X:
      max_prob = 0
      winning_class = -1

      for c in self.classes:
        index = [i for i, x_ in enumerate(x) if x_ == 1]
        prob = 1
        for i in index:
          if i in self.condprob and self.condprob[i][c] != 0:
            prob *= self.condprob[i][c]
          else:
            pass
      
        prob = (prob * self.prior[c])

      
        if prob > max_prob:
          max_prob = prob
          winning_class = c
      
      results.append(winning_class)

    return results




      
  
def train_test_split(X, y, where=0.8):
  chop = int(len(X) * 0.8)
  train_X = X[:chop]
  train_y = y[:chop]

  test_X = X[chop:]
  test_y = y[chop:]

  return train_X, train_y, test_X, test_y


def f_measure(predictor, X, y):

  true_positive = 0
  false_positive = 0
  true_negative = 0
  false_negative = 0

  for i in range(len(X)):
    prediction = predictor.predict([X[i]])[0]

    if y[i] == 1:
      if prediction == 1:
        true_positive += 1
      else:
        false_positive += 1
    elif y[i] == 0:
      if prediction == 0:
        true_negative += 1
      else:
        false_negative += 1
  
  try:
    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)
  except:
    return np.nan

  if true_positive == 0:
    return 0
  return (2 * precision * recall) / (precision + recall)


def main():

  bodies = pd.read_csv('dbworld_bodies_stemmed.csv', index_col=0).values
  subjects = pd.read_csv('dbworld_subjects_stemmed.csv', index_col=0).values
  tests = pd.read_csv("test.csv", index_col=0).values

  test_values = tests[:, :-1]
  test_classes = tests[:, -1]

  body_values = bodies[:, :-1]
  subject_values = subjects[:,:-1]

  body_classes = bodies[:, -1]
  subject_classes = subjects[:, -1]

  test_train_X, test_train_y, test_test_X, test_test_y = train_test_split(test_values, test_classes)
  body_train_X, body_train_y, body_test_X, body_test_y = train_test_split(body_values, body_classes)
  subject_train_X, subject_train_y, subject_test_X, subject_test_y = train_test_split(subject_values, subject_classes)
  

  print("----------------- Body Results ---------------------")
  body_clf = NaiveBayes()
  body_clf.fit(body_train_X, body_train_y)
  # print("Custom NB Body Score:", body_clf.score(body_test_X, body_test_y))
  print("Custom NB F-Measure:", f_measure(body_clf, body_test_X, body_test_y))


  clf_sk  = MultinomialNB()
  clf_sk.fit(body_train_X, body_train_y)
  # print("Multinomial NB Body Score:", clf_sk.score(body_test_X, body_test_y))
  print("Multinomial NB F-Measure:", f_measure(clf_sk, body_test_X, body_test_y))


  print("----------------- Subject Results -------------------")

  subject_clf = NaiveBayes()
  subject_clf.fit(subject_train_X, subject_train_y)
  # print("Custom NB Subject Score:", subject_clf.score(subject_test_X, subject_test_y))
  print("Custom NB F-Measure:", f_measure(subject_clf, subject_test_X, subject_test_y))


  sub_clf_sk = MultinomialNB()
  sub_clf_sk.fit(subject_train_X, subject_train_y)
  # print("Multinomial NB Subject Score:", sub_clf_sk.score(subject_test_X, subject_test_y))
  print("Multinomial NB F-Measure:", f_measure(sub_clf_sk, subject_test_X, subject_test_y))


  print("------------------ Simple Test File -------------------")
  test_clf = NaiveBayes()
  test_clf.fit(test_train_X, test_train_y)
  print("Custom NB F-Measure:", f_measure(test_clf, test_test_X, test_test_y))

  test_clf_sk = NaiveBayes()
  test_clf_sk.fit(test_train_X, test_train_y)
  print("Multinomial NB F-Measure:", f_measure(test_clf_sk, test_test_X, test_test_y))




if __name__ == "__main__":
  main()