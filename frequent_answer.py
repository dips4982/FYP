

import json
import pickle
import operator
from collections import Counter

frequent_ans={} # dict to store frequency of each answer
ans_file=open("/content/drive/MyDrive/frequent_answer/pretty_ans.json") # Open the answer file
data_ans=json.load(ans_file) #Load the json file
count=0

"""### Iterating over file and finding the frequency of each answer

"""

count=0
frequent_ans={}
for i in data_ans['annotations']:
    count+=1
    if i['multiple_choice_answer'] in frequent_ans:
      frequent_ans[i['multiple_choice_answer']]+=1
    else:
      frequent_ans[i['multiple_choice_answer']]=0
# Closing file
print(frequent_ans)
ans_file.close()

"""### Sorting the Dictonary

"""

frequent_ans_sorted = dict(sorted(frequent_ans.items(),key=operator.itemgetter(1),reverse=True))

"""### Finding most frequent 3000 answers

"""

count_ans = Counter(frequent_ans_sorted)
most_common_ans = dict(count_ans.most_common(3000))
print(most_common_ans)

"""### Storing the answer in file

"""

try:
    frequent_answers_file = open('frequent_answers', 'wb')
    pickle.dump(most_common, frequent_answers_file)
    frequent_answers_file.close()
  
except:
    print("Something went wrong")