import numpy
import csv
import random
import time
class iterativeCF:
    def load_movies_data(self,ratings_file,fraction,seed=time.time()): #fration is the proportion of input data to be in the test set
        record={} #test set
        userSet = set()  #for obtaining users matrix dimension
        datas=[]  #training set
        Mcounter=0
        mapMIdToMovie={}  #map movieId to movie_index (reducing matrix size)
        mapMovieToMId={}   #map movie_index to movieId
        numberOfMRating=0    #number of rating
        sumRating=0          #rating accumulator
        random.seed(seed)
        with open(ratings_file, 'r', encoding='ISO-8859-1') as csvR:
            CSVR = csv.reader(csvR)
            for row in CSVR:
                if not row[0].isdigit(): continue
                movieID = int(row[1])
                userID = int(row[0])
                rating=float(row[2])
                if movieID not in mapMIdToMovie:
                    mapMIdToMovie[movieID]=Mcounter
                    mapMovieToMId[Mcounter]=movieID
                    Mcounter+=1
                if userID not in userSet:
                    userSet.add(userID)
                if random.random() >= fraction:  #put this data to training set
                    datas.append((userID,mapMIdToMovie[movieID],rating))
                    numberOfMRating += 1
                    sumRating += rating
                else:                           #put this data to test set
                    record[(userID, mapMIdToMovie[movieID])] = float(row[2])
        return sumRating/numberOfMRating, datas,userSet,mapMIdToMovie,mapMovieToMId,record

    def __init__(self,fileName,fration):
        mean,datas, userSet, mapMIdToMovie, mapMovieToMId, missData = self.load_movies_data(fileName, fration)
        self.datas=datas
        self.userSet=userSet
        self.mapMIdToMovie=mapMIdToMovie
        self.mapMovieToMId=mapMovieToMId
        self.missData=missData
        self.mean=mean
        D_in = len(self.userSet)
        D_out = len(self.mapMIdToMovie)
        innerDim = 100  #inner dimension of matrix
        self.w=numpy.random.rand(D_in, innerDim)  #create user matrix w
        self.h=numpy.random.rand(innerDim, D_out)   #create movie matrix h
        self.bu=numpy.zeros(D_in)  #user bias
        self.bi=numpy.zeros(D_out)  #movie bias

    def seeRating(self,usr,mId):  #expected rating comprise multiplication of submtrices of w and h, and mean of whole rating and user bias,movie bias
        return numpy.matmul(self.w[usr-1:usr,:],self.h[:,self.mapMIdToMovie[mId]:self.mapMIdToMovie[mId]+1]).item()+self.mean+self.bu[usr-1]+self.bi[self.mapMIdToMovie[mId]]

    def seePerformace(self):  #loop through the test set and compare them with expected results
        summation=0
        counter=0
        for ele in self.missData:
            expected=numpy.matmul(self.w[ele[0]-1:ele[0],:],(self.h[:,ele[1]:ele[1]+1])).item()+self.mean+self.bu[ele[0]-1]+self.bi[ele[1]]
            summation+=(self.missData[ele]-expected)**2
            counter+=1
            print("( usr:",ele[0],"mid:",self.mapMovieToMId[ele[1]],"rating:",self.missData[ele],") expected_rating:",expected)
        return (summation/counter)**0.5  #compute the RMSE

    def StartTraining(self):
        print("training started!")
        suf = [i for i in range(len(self.datas))]  #indices of training set
        lamda = 0.09  #regularisation factor
        gamma = 0.02  #learning rate
        for t in range(100): #iterate 100 times
            random.shuffle(suf)  #shuffling indices of training set
            print("iteration:", t)
            for index in suf:
                W_index = self.datas[index][0]
                Wtemp = self.w[W_index - 1:W_index, :]  #get the wu from matrix w
                H_index = self.datas[index][1]
                Htemp = self.h[:, H_index:H_index + 1]  #get the hi from matrix h
                expected_rating=max(min(numpy.matmul(Wtemp,Htemp).item()+self.mean+self.bu[W_index - 1]+self.bi[H_index],5),0)  #clamp the expected rating between 0 to 5 inclusively
                eui = self.datas[index][2] - expected_rating
                Wr = Wtemp + gamma * (eui * Htemp.transpose() - lamda * Wtemp)  #update wu
                Hr = Htemp + gamma * (eui * Wtemp.transpose() - lamda * Htemp)  #update hi
                self.w[W_index - 1:W_index, :] = Wr   #feed updated wu back to w
                self.h[:, H_index:H_index + 1] = Hr   #feed updated hi back to h
                self.bu[W_index - 1]=self.bu[W_index - 1]+gamma*(eui-lamda*self.bu[W_index - 1]) #update user bias
                self.bi[H_index]=self.bi[H_index]+gamma*(eui-lamda*self.bi[H_index]) #update movie bias
        print("training ended!")


start=time.time()
cf=iterativeCF("ratings.csv",0.2)
cf.StartTraining()
print("RMSE:",cf.seePerformace())
print("time_used:",time.time()-start)
