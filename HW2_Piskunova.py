import nltk
from nltk.collocations import *
from nltk.metrics.spearman import *
import csv

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

infile = open('court-V-N.csv', 'r', encoding = 'utf-8')
table = []
words_tagged = []
for row in csv.reader(infile):
    table.append(row)
for row in table:
    for word in row:
        word_low = word.lower()
        words_tagged.append(word_low)
        #print(word_low)
finder = BigramCollocationFinder.from_words(words_tagged)
finder.apply_freq_filter(3)
#print(finder.nbest(bigram_measures.pmi, 1000))
scored_bigrams = finder.score_ngrams(bigram_measures.pmi)
#for i in sorted(scored_bigrams, key = lambda x: x[1]):
#    print(i)
#print(sorted(scored_bigrams, key = lambda x: x[1]))
scored_bigrams2 = finder.score_ngrams(bigram_measures.chi_sq)
for i in sorted(scored_bigrams2, key = lambda x: x[1]):
    print(i)
gold_standart = [('рассматривать', 'дело'),
                 ('принять', 'решение'),
                 ('вынести', 'приговор'),
                 ('вынести', 'решение'),
                 ('вынести', 'постановление'),
                 ('удовлетворить', 'иск'),
                 ('рассматривать', 'иск'),
                 ('отклонить', 'ходатайство'),
                 ('признать', 'виновным'),
                 ('принять', 'внимание')]
results_list = [('рассматривать', 'иск'),
                ('отклонить', 'ходатайство'),
                ('вынести', 'решение'),
                ('рассматривать', 'дело'),
                ('удовлетворить', 'иск'),
                ('вынести', 'приговор'),
                ('принять', 'решение'),
                ('признать', 'виновным'),
                ('принять', 'внимание'),
                ('вынести', 'постановление')] #pmi
results_list2 = [('отклонить', 'ходатайство'),
                ('рассматривать', 'дело'),
                ('принять', 'внимание'),
                ('вынести', 'постановление'),
                ('вынести', 'приговор'),
                ('рассматривать', 'иск'),
                ('вынести', 'решение'),
                ('признать', 'виновным'),
                ('принять', 'решение'),
                ('удовлетворить', 'иск')] #liklihood
print('%0.1f' % spearman_correlation(ranks_from_sequence(gold_standart), ranks_from_sequence(results_list)))#pmi (0.1)
print('%0.1f' % spearman_correlation(ranks_from_sequence(gold_standart), ranks_from_sequence(results_list2)))#liklihood (-0.1)


infile.close()

#Для аналииза были взяты две меры: pmi и log liklihood. При сравнении результатов с золотым стандартом мы воспользовались коэффициентом ранговой
#корреляции Спирмена. В итоге для меры pmi корреляция получилась 0.1, что является не очень хорошим результатом, так как списки почти не совпадают с золотым стандартом.
#Но с мерой log liklihood коэффициент корреляции получился еще меньше, а именно -0.1, что означает, что процент совпадений с золотым стандартом еще меньше. 
