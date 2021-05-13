# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 13:54:19 2019

@author: 赵怀菩
"""
import bst
import distance
class base_dpc(object):
    """
    a basic dpc, will use by ldpc and gdpc(succeed)
    """
    def __init__(self,**kwargs):
        self.X=kwargs['X']
        self.dcfactor=0
        self.rouc=0
        self.deltac=0
        self.gammac=0
        self.rouo=0
        self.deltao=0
        try:
            self.dcfactor=kwargs['dcfactor']
            self.rouc=kwargs['rouc']
            self.deltac=kwargs['deltac']
            self.gammac=kwargs['gammac']
            self.rouo=kwargs['rouo']
            self.deltao=kwargs['deltao']
        except KeyError as ae:
            print ('key error',ae)
        self.cal_dc()
        self.cal_rou()
        self.cal_delta()
        self.pick_cluster()
        self.pick_outlier()
    
    def cal_distance(self):
        #deprecated.
        #will not calculate all distance at the same time
        #use cal_dc, then use cal_rou and cal_delta directly
        self.distance=[]
        self.cal_dc_index()
        for i,x in enumerate(self.X):
            for j,y in enumerate(self.X[i:]):
                #distance function can be l2 cosine, or else.
                #recommand be simple
                #distance=distance(x,y)
                distance=0
                self.distance.append((i,j),distance)
                pass
            
    def cal_dc_index(self):
        if self.dcfactor <= 0 or self.dcfactor > 1:
            self.dcfactor = 0.02
        assume_distance_number = (len(self.X) * (len(self.X)+1)) / 2
        assume_distance_number = int(assume_distance_number)
        self.dc_index = int(assume_distance_number*self.dcfactor)
        if self.dc_index <= 0:
            print('caution that dc index is 0.\n means that each point is single cluster.')
        elif self.dc_index >= assume_distance_number:
            print('caution that dc index is length of distance number.\n means that each point is in one cluster.')
        pass

    def cal_dc(self):
        #if self.dcfactor<=0 or self.dcfactor>1:
        self.cal_dc_index()
        #distance_list=sorted(self.distance,key=lambda x:x[1])
        bst_list=[]
        self.distance_bst=None
        
        for i,x in enumerate(self.X):
            #caution that i != j avoid the distance 0
            for j,y in enumerate(self.X[i+1:]):
                #distance function can be l2 cosine, or else.
                #recommand be simple
                #distance=distance(x,y)
                dis=distance.distance(x,y)
                if len(bst_list) <= self.dc_index:
                    bst_list.append(dis)
                    if len(bst_list) > self.dc_index:
                        #init the bst
                        self.distance_bst=bst.BST(bst_list)
                        #self.distance_bst.inOrderTraverse(self.distance_bst.root)
                else:
                    n,p=self.distance_bst.get_rightmost()
                    if n.data <= dis:
                        continue
                    else:
                        #print (' ')
                        #print ('del',n.data,'add',dis)
                        self.distance_bst.delete_node(n,p)
                        self.distance_bst.insert(dis)
                        
                        #self.distance_bst.inOrderTraverse(self.distance_bst.root)
                pass
            
        n,p=self.distance_bst.get_rightmost()
        self.dc=n.data
            
    def cal_rou(self):
        
        pass
    def cal_delta(self):
        pass
    def pick_cluster(self):
        pass
    def pick_outlier(self):
        pass
    pass

if __name__=='__main__':
    d=[49, 38, 65, 97, 60, 76, 13, 27, 5, 1]
    a=base_dpc(X=d,dcfactor=0.1,rouc=1,deltac=1,gammac=1,rouo=1,deltao=1)
    print ('dc',a.dc)
    del a
    pass
