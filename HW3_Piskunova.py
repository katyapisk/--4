import re
import numpy as np
from sklearn import model_selection, svm
from matplotlib import mlab
from matplotlib import pyplot as plt

with open('anna.txt', encoding='utf-8') as f:
    anna = f.read()
with open('sonets.txt', encoding='utf-8') as f:
    sonets = f.read()

anna_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', anna)
sonet_sentences = re.split(r'(?:[.]\s*){3}|[.?!]', sonets)

def vowel(word):
    vowels = ['а', 'у', 'о', 'ы', 'и', 'э', 'я', 'ю', 'ё', 'е']
    n = 0
    for letter in word:
        if letter in vowels:
            n += 1
    return n

def diff_letters(words):
    diff_letter = set()
    for word in words:
        word = word.strip('.,!;:-()""—')
        for letter in word:
            diff_letter.add(letter)
    return len(diff_letter)

def features(c, corpus):
    features = []
    for line in corpus:
        line = line.strip()
        line = line.lower()
        words = line.split(' ')
        if len(line) > 0:
            differ = diff_letters(words)
            len_let = [len(word.strip('.,?!"\/;:*-()—').lstrip('.,?!"\/;:*-()—')) for word in words if len(word) > 0]
            vowels  = [vowel(word) for word in line if len(word) > 0]
            if len(len_let) > 0:
                features.append([c, np.sum(len_let), differ, np.sum(vowels), np.median(len_let), np.median(vowels)])
    return features

anna_data = features(1, anna_sentences)
sonet_data = features(2, sonet_sentences)
anna_data = np.array(anna_data)
sonet_data = np.array(sonet_data)
data = np.vstack((anna_data, sonet_data))
p = mlab.PCA(data[:, 1:], True)
N  = len(anna_data)
plt.figure()
plt.plot(p.Y[:N,0], p.Y[:N,1], 'og', p.Y[N:,0], p.Y[N:,1], 'sb')
plt.show()
clf = svm.LinearSVC(C=0.1)
clf.fit(data[::2, 1:], data[::2, 0])
print(clf.score(data[::2, 1:], data[::2, 0]))
m = 0
for i in data[1::2, :]:
    label = clf.predict(i[1:].reshape(1, -1))
    if label != i[0] and m < 3:
        print('Ошибка: class = ', i[0], ', label = ', label, ', экземпляр ', i[1:])
        m += 1
    if m > 3:
        break

    
                
