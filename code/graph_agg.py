import matplotlib.pyplot as plt
import numpy as np


def make_bar_data(list1):
    bar_data = []
    bar_data.append(list1[0])
    running_sum = list1[0]
    for i in range(1,len(list1)):
        bar_data.append(list1[i] - running_sum)
        running_sum = list1[i]
    return bar_data



top_15_results = {3: [21.909   ,32.733   ,36.744   ,40.720   ,43.707   ,46.693   ,49.358   ,51.498   ,53.590   ,55.248   ,56.954   ,58.598   ,60.150   ,61.424],
4: [23.128   ,32.660   ,37.950   ,41.593   ,45.189   ,48.255   ,51.072   ,53.731   ,56.061   ,57.661   ,59.314   ,60.995   ,62.260   ,63.350],
5: [20.426   ,31.181   ,36.573   ,40.902   ,44.428   ,47.312   ,50.131   ,52.686   ,54.566   ,56.412   ,57.868   ,59.170   ,60.470   ,61.591],
6: [24.628   ,34.710   ,39.734   ,43.301   ,46.302   ,49.066   ,51.389   ,53.411   ,55.314   ,56.791   ,58.410   ,59.799   ,60.910   ,62.073],
7: [22.933   ,34.401   ,40.308   ,44.083   ,46.749   ,49.419   ,51.965   ,54.051   ,55.785   ,57.461   ,59.073   ,60.470   ,62.034   ,63.265],
8:[21.798   ,34.137   ,39.495   ,43.242   ,46.306   ,48.875   ,51.014   ,53.472   ,55.400   ,57.050   ,58.899   ,60.570   ,61.789   ,63.034 ],
9: [23.013   ,33.964   ,38.750   ,42.566   ,45.765   ,48.243   ,50.718   ,53.059   ,55.310   ,57.523   ,59.487   ,60.866   ,62.195   ,63.482],
10 :[23.454   ,33.937   ,38.560   ,42.161   ,45.273   ,48.020   ,50.294   ,52.394   ,54.212   ,55.974   ,57.194   ,58.503   ,59.887   ,61.202]
}
MAX_RANK = 11
att_avrg = np.zeros(len(top_15_results)*10).reshape(len(top_15_results),10)
for i, key in enumerate(top_15_results):
    att_avrg[i] = top_15_results[key][:10]
bar_avrg = np.zeros(len(top_15_results)*10).reshape(len(top_15_results),10)
for i, key in enumerate(top_15_results):
    bar_avrg[i] = make_bar_data(top_15_results[key][:10])

nb_results = [15.149,20.423,23.347,25.922,27.759,29.057,30.335,31.187,31.995,32.843,33.516,34.179,34.693,35.138][:10]
bar_nb_data = make_bar_data(nb_results)
tfidf_results = [19.218   ,25.237   ,29.284   ,32.097   ,34.284   ,36.114   ,37.669   ,39.023   ,40.079   ,40.961   ,41.824   ,42.602   ,43.255   ,43.893 ][:10]
bar_tf_data = make_bar_data(tfidf_results)
cbow_results = [16.333 ,25.362 ,29.850 ,33.038 ,35.402 ,37.362 ,39.067 ,40.939 ,42.406 ,43.659 ,44.800 ,46.126 ,47.168 ,48.190][:10]
bar_cbow_data = make_bar_data(cbow_results)


# plt.errorbar(x=RANKS, y=np.mean(nb_avrg,0), yerr=np.std(nb_avrg,0), label='Naive Bayes',linewidth=2, color='blue')
# plt.plot(nb_avrg/len(folds),label='Naive Bayes',linewidth=2)
# plt.axvline(x= np.mean(np.mean(nb_avrg,0)),linestyle='--', color='blue')

# plt.fill_between(list(range(0,MAX_RANK -1)), np.mean(nb_avrg,0) - np.std(nb_avrg,0), np.mean(nb_avrg,0) + np.std(nb_avrg,0) ,alpha=0.3, facecolor='b')
# plt.plot(tfidf_avrg/len(folds),label='Tf-idf cosine sim',linewidth=2)
# plt.errorbar(x=RANKS,y=np.mean(tfidf_avrg,0), yerr=np.std(tfidf_avrg,0), label='Tf-idf cosine sim',linewidth=2, color='green')
# plt.axvline(x= np.mean(np.mean(tfidf_avrg,0)),linestyle='--', color='green')
plt.title('Accuracy in Top N ranks'.format(0))
plt.ylabel('Accuracy')
plt.xlabel('Top N')
plt.ylim(0,60)
plt.plot(list(range(1,MAX_RANK)),nb_results,label='Naive Bayes',linewidth=2, color='b')
plt.plot(list(range(1,MAX_RANK)),tfidf_results,label='Tf-idf cosine sim',linewidth=2, color='g')
plt.plot(list(range(1,MAX_RANK)),np.mean(att_avrg,0),label='EDA',linewidth=2, color='r')
plt.fill_between(list(range(1,MAX_RANK)), np.mean(att_avrg,0) - np.std(att_avrg,0), np.mean(att_avrg,0) + np.std(att_avrg,0) ,alpha=0.3, facecolor='r')
plt.plot(list(range(1,MAX_RANK)),cbow_results,label='CBOW cosine sim',linewidth=2, color='orange')
plt.legend(loc= 4)
plt.show()
# plt.fill_between(list(range(0,MAX_RANK -1)), np.mean(tfidf_avrg,0) - np.std(tfidf_avrg,0), np.mean(tfidf_avrg,0) + np.std(tfidf_avrg,0) ,alpha=0.3, facecolor='g')




# plt.plot(att_avrg/len(folds),label='Decomposable Attention',linewidth=2)
# plt.errorbar(x=RANKS,y=np.mean(att_avrg,0), yerr=np.std(att_avrg,0), label='Decomposable Attention',linewidth=2, color='red')
# plt.axvline(x= np.mean(np.mean(att_avrg,0)),linestyle='--', color='red')



# plt.plot(np.mean(lda_avrg,0),label='LDA cosine sim',linewidth=2)


# print([bar_nb_data/len(folds),bar_tf_data/len(folds),bar_da_data/len(folds)])
plt.title('Accuracy in each Rank')
plt.ylabel('Accuracy')
plt.xlabel('Rank')
xx = np.asarray(range(0,MAX_RANK-1))
plt.bar(xx, bar_nb_data, width=0.2, facecolor='b', edgecolor='b', linewidth=3, alpha=.5, label='Naive Bayes')
plt.bar(xx+0.2, bar_cbow_data, width=0.2, facecolor='orange', edgecolor='orange', linewidth=3, alpha=.5, label='CBOW cosine')
plt.bar(xx+0.4, bar_tf_data, width=0.2, facecolor='g', edgecolor='g', linewidth=3, alpha=.5, label='Tf-idf cosine')
plt.bar(xx+0.6, np.mean(bar_avrg,0), width=0.2, facecolor='r', edgecolor='r', linewidth=3, alpha=.5, label='EDA',yerr=np.std(bar_avrg,0),ecolor='black')
plt.xticks(xx+0.35, range(1,MAX_RANK))
plt.legend()
plt.show()

print("Naive: {}".format(bar_nb_data))
print("Tfidf: {}".format(bar_tf_data))
print("CBOW: {}".format(bar_cbow_data))
print("EDA: {}".format(np.mean(bar_avrg,0)))

print(np.std(bar_avrg,0))

# plt.set_xticks(xx + 0.6 / 2)
# plt.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9', '10'))
# plt.bar(xx, bar_nb_data, width=0.2, facecolor='b', edgecolor='b', linewidth=3,  label='Naive Bayes')
# plt.bar(xx+0.2, bar_cbow_data, width=0.2, facecolor='orange', edgecolor='orange', linewidth=1,  label='CBOW Cosine Sim')
# plt.bar(xx+0.4, bar_tf_data, width=0.2, facecolor='g', edgecolor='g', linewidth=1,  label='Tf-idf Cosine Sim')
# plt.bar(xx+0.6, np.mean(bar_avrg,0), width=0.2, facecolor='r', edgecolor='r', linewidth=1,  label='Decomposable Attention',yerr=np.std(bar_avrg,0),ecolor='black')

