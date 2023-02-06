import numpy as np
from scipy import stats
  		  	   		  	  		  		  		    	 		 		   		 		  
class DecisionTree(object):	  	   		  	  		  		  		    	 		 		   		 		  
    def __init__(self, leaf_size):  		  	   		  	  		  		  		    	 		 		   		 		  
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        Constructor method: initialize leaf size
        """  		  	   		  	  		  		  		    	 		 		   		 		  
        self.leaf_size=leaf_size	   		  	  		  		  		    	 		 		   		 		  	  
  		  	   				 		   		 		  
    def add_evidence(self, data_x, data_y):  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        To train the learner
        $param data_x: A series of feature values used for training the learner
        $type data_x: numpy.ndarray
        $param data_y: the label of the data, which is the goal for the learner to predict
        $type data_y: numpy.ndarray
        """

        #Make sure the data is in right shape
        data_y=np.reshape(data_y,(data_y.shape[0],1))
        #Concatenate the data
        data=np.concatenate((data_x,data_y),axis=1)
        #Initialize the decision tree
        self.trees=np.empty((0,4),float)
        #Recursive function to build the decision tree
        def build_tree(data):
            #Base case to stop further recursion, ending in a leaf node with mode
            if data.shape[0] <= self.leaf_size:
                return np.array([[-1,stats.mode(data[:,-1])[0],0,0]])

            #Identify the best feature and its median value to split data on
            idx,split_val=self.best_split(data)

            #If remaining data has the same value or when split is not possible (idx==-1)
            #End the recursion in a leaf node
            if np.all(data[:,idx]==data[0,idx]) or idx==-1:
                return np.array([[-1,stats.mode(data[:,-1])[0],0,0]])

            #Build the left tree first
            left_tree=build_tree(data[data[:,idx]<=split_val])
            #Then build the right tree
            right_tree=build_tree(data[data[:,idx]>split_val])
            #Create the root array
            root=np.array([[idx,split_val,1,left_tree.shape[0]+1]])
            #Concatenate the arrays to form a decision tree table
            self.trees=np.concatenate((root,left_tree,right_tree),axis=0)
            return self.trees
        
        self.trees=build_tree(data)
       
    def best_split(self, data):
        #get the list of highest absolute correlation, sorted in descending order
        idx_list=self.highest_abs_corr(data)
        pos=0
        idx = idx_list[pos]
        split_val = np.median(data[:, idx])
        #to ensure split is possible, if not then keep finding the next highest correlated index
        while split_val==np.max(data[:,idx]) and (data.shape[1]-(pos+1))> 1:
            pos+=1
            idx = idx_list[pos]
            split_val = np.median(data[:, idx])
        if (data.shape[1]-(pos+1))== 1: #only y column left
            return -1, -1
        return idx, split_val

    #return the list of highest absolute correlation, sorted in descending order
    def highest_abs_corr(self, data):
        temp=np.corrcoef(data,rowvar=False)
        temp_list=np.absolute(temp[-1,:-1])*(-1)
        idx_list=np.argsort(temp_list)

        return idx_list
    
    def query(self, points):  		  	   		  	  		  		  		    	 		 		   		 		  
        """
        To predict the label, Y given a series of feature X
        $param points: A numpy array of feature X
        $type points: numpy.ndarray
        $return: A numpy array of predicted label, Y according to feature X
        $rtype: numpy.ndarray
        """
        #Recursively access the decision tree to obtain the predicted classification
        def query_tree(data,start_row=0):
            idx=np.int(self.trees[start_row,0])

            #we define leaf node to have idx = -1
            if idx==-1:
                return self.trees[start_row,1]
            #if idx is not -1, continue searching until idx reaches a leaf node
            elif data[idx]<=self.trees[start_row,1]: #search left node
                return query_tree(data,np.int(start_row+self.trees[start_row,2]))
            else: #search right node
                return query_tree(data,np.int(start_row+self.trees[start_row,3]))

        #store query results in a numpy array
        self.query_result=np.empty((points.shape[0]),float)
        #loop through the input data, and store each prediction in self.query_result
        for i,point in enumerate(points):
            temp=query_tree(point)
            self.query_result[i]=temp
        return self.query_result