import numpy as np

class BootstrapAggregating(object):
    def __init__(self,learner, kwargs, bags = 20):
        """
        Constructor method: initialize type of learner and number of bags
        """
        #Define the type of learner we want to use
        self.learner=learner
        #Arguments for learner
        self.kwargs=kwargs
        #Number of learners
        self.bags=bags

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

        #Store learners in an array
        self.learners=[]
        for i in range(self.bags):
            self.learners.append(self.learner(**self.kwargs))
        for learner in self.learners:
            #Bootstrap data
            bag_data = data[np.random.randint(data.shape[0], size=data.shape[0]), :]
            bag_data_x = bag_data[:, :-1]
            bag_data_y = bag_data[:, -1]
            learner.add_evidence(bag_data_x, bag_data_y)

    def query(self, points):
        #store query results in a numpy array
        self.query_result=np.zeros((points.shape[0]),float)
        #for each learner, get the prediction result and calculate the mean of it
        for learner in self.learners:
            result=learner.query(points)
            self.query_result=np.sum((result,self.query_result),axis=0)
        self.query_result=self.query_result/self.bags

        return self.query_result