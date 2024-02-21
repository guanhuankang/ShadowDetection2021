import os
import numpy as np
from PIL import Image
import progressbar

class Evaluation:
    def __init__(self, _gtPath, _evalPath, _name="unknow"):
        self.gp = _gtPath
        self.ep = _evalPath
        self.name = _name
        self.ans = None
    
    def echo(self):
        if isinstance(self.ans, type(None)):return None
        ech = "\nBER:%f\nshadow Ber:%f\nnon-shadow BER:%f\naccuracy:%f\n"%(\
            self.ber, self.shadowBer, self.nonshadowBer, self.accuracy)
        print(ech)
        return ech

    
    def check(self, gL, eL):
        tot = len(gL)
        for i in range(tot):
            if gL[i][0:-4]!=eL[i][0:-4]:return False
        return True
    
    def calc(self):
        gL, eL = os.listdir(self.gp), os.listdir(self.ep)
        gL.sort()
        eL.sort()

        if not self.check(gL, eL):
            raise "The files in the given folder(GT and Results) do not match! \
                 Please check the path of GT and results"
            return False
        
        ret = np.zeros(4, dtype=int) ## TP,TN,FP,FN
        tot = len(gL)
        widgets = [progressbar.Percentage(),progressbar.Timer(),progressbar.Bar(),progressbar.ETA()]
        bar = progressbar.ProgressBar(widgets=widgets,maxval=tot).start()
        for i in range(tot):
            _gt = np.array(Image.open(os.path.join(self.gp, gL[i])).convert("L")).astype("f")
            gt = (np.sign(_gt-127.5)/2+1.0).astype(int) ## -> 0 or 1
            _et = np.array(Image.open(os.path.join(self.ep, eL[i])).convert("L")).astype("f")
            et = (np.sign(_et-127.5)/2+1.0).astype(int) ## -> 0 or 1

            gtsum = gt.shape[0]*gt.shape[1]
            p = gt.sum()
            n = gtsum - p

            fn = ((gt-et)==1).sum()
            fp = ((et-gt)==1).sum()
            tp = p - fn
            tn = n - fp

            ret += np.array([tp,tn,fp,fn])

            bar.update(i+1)

        self.ans = ret
        self.ber = self.getBER()
        self.shadowBer = self.getShadowBer()
        self.nonshadowBer = self.getNonshadowBer()
        self.accuracy = self.getAccuracy()
        return True
    
    def getBER(self):
        if isinstance(self.ans, type(None)):return None
        tp,tn,fp,fn = self.ans
        return 100*(1. - 0.5 * (tp/(tp+fn)+tn/(tn+fp)))
    
    def getShadowBer(self):
        if isinstance(self.ans, type(None)):return None
        tp,tn,fp,fn = self.ans
        return 100*( 1. - tp/(tp+fn) )
    
    def getNonshadowBer(self):
        if isinstance(self.ans, type(None)):return None
        tp,tn,fp,fn = self.ans
        return 100*( 1. - tn/(tn+fp) )
    
    def getAccuracy(self):
        if isinstance(self.ans, type(None)):return None
        tp,tn,fp,fn = self.ans
        return 100*( (tp+tn)/self.ans.sum() )

# from config import evalPath, gtPath, name
#
# evaluation = Evaluation(gtPath, evalPath, name)
# evaluation.calc()
# evaluation.echo()
# print(evaluation.ans.sum())