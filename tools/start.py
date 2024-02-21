import os
from evaluation import Evaluation
from config import evalPath, gtPath, name


evaluation = Evaluation(gtPath, evalPath, name)
evaluation.calc()
ech = evaluation.echo()
print("tot:",evaluation.ans.sum())

if not os.path.exists("results"):
	os.mkdir("results")
	
with open("results/log_%s.ans"%(evaluation.name,), "w") as f:
    f.write( ech+"\nTP:%d,TN:%d,FP:%d,FN:%d"%tuple(evaluation.ans) )

with open("results_history.txt","a+") as f:
    f.write(evaluation.name+"\n"+ech+"-----------------------------\n")