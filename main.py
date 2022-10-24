from datetime import datetime, timedelta
from operator import indexOf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
import csv


####################
#SET UP DATA ARRAYS#
####################

######################
#READ INSULINDATA.CSV#
######################

insulinDF = pd.read_csv('InsulinData.csv', usecols = ['Date', 'Time', 'BWZ Carb Input (grams)'])
insulinDF.rename(columns = {'BWZ Carb Input (grams)': 'Carbs'}, inplace=True)
#print(insulinDF)
insulinDF = insulinDF.iloc[::-1] #reverse insulinDF
insulinDF = insulinDF.reset_index()
insulinDF.drop('index', inplace=True, axis=1)
#insulinDF = insulinDF.reindex(index=insulinDF.index[::-1])
#print(insulinDF)


##Add all mealtimes to mealTimes array
##Also add carbs into carbGrams array
mealTimes = []
carbGrams = []
for idx, row in insulinDF.iterrows():
    if not pd.isnull(row['Carbs']) and row['Carbs'] != '0':
        time = datetime.strptime(row['Date']+" "+row['Time'], '%m/%d/%Y %H:%M:%S')
        mealTimes.append(time)
        carbGrams.append(row['Carbs'])

##Filter through mealTimes and delete those that have a meal less than 2 hours after that time
##Also delete carbGrams in same index
j = 0
for i in range(len(mealTimes) - 1):
    #print('Comparing: ', mealTimes[j], 'to: ', mealTimes[j+1])

    while(mealTimes[j] >= (mealTimes[j+1] - timedelta(hours = 2))):
        #print('Comparing: ', mealTimes[j], 'to 2-hours before: ', (mealTimes[j+1] - timedelta(hours = 2)))
        #print('Deleted mealTimes[j]: ', mealTimes[j])
        del mealTimes[j]
        del carbGrams[j]

    while(carbGrams[j] == 0):
        del mealTimes[j]
        del carbGrams[j]
    
    j += 1

    #print('j ', j, 'len(mealTimes)', len(mealTimes))
    if j == (len(mealTimes) - 1):
        mealTimes.append(datetime.now()) ##Add last mealTime so future loop doesn't index out of range
        break

########################
#CREATE AND ASSIGN BINS#
########################
##Bins are ground truth for this assignment
carbMin = min(carbGrams)
carbMax = max(carbGrams)
numBins = int((carbMax - carbMin) / 20) + 1 #int() always rounds down, need to add 1

carbBin = []

for i in carbGrams:
    carbBin.append(int((i - carbMin) / 20))



########################
#    READ CGMDATA.CSV  #
# CREATE MEALDATAARRAY #
########################
mealDataArray = []

cgmDF = pd.read_csv('CGMData.csv', usecols = {'Date', 'Time', 'Sensor Glucose (mg/dL)'})
cgmDF.rename(columns = {'Sensor Glucose (mg/dL)': 'Glucose'}, inplace=True)
cgmDF = cgmDF.iloc[::-1] #reverse insulinDF
cgmDF = cgmDF.reset_index()
cgmDF.drop('index', inplace=True, axis=1)
#print(cgmDF)

mealTimeIndex = 0

i = 0
while i < (len(cgmDF) - 30):
    row = cgmDF.iloc[i]
    time = datetime.strptime(row['Date']+" "+row['Time'], '%m/%d/%Y %H:%M:%S')

    #print('Enter big while loop')

    if time >= mealTimes[mealTimeIndex] - timedelta(hours = 0.5):
        #print('Add to mealDataArray, time:', time)
        mealData = []
        fullArray = 0

        for j in range(30):

            if pd.isna(row['Glucose']):
                i += 1
                break #for now, if there's a NaN, skip that array
                #print('null row[glucose]:', row['Glucose'])
                mealData.append(0)
            else:
                #print('not null row[glucose]:', row['Glucose'])
                mealData.append(int(row['Glucose']))
            
            i+= 1
            row = cgmDF.iloc[i]

            if j == 29:
                #print('Full Array 1')
                fullArray = 1
        
        #print('mealData: ', mealData)
        if fullArray:
            mealDataArray.append(mealData)
            i -= 6 #move back 30 minutes before searching for next meal time

        mealTimeIndex += 1 
    
    else:
        i += 1

#print('noMealDataArray:', noMealDataArray)

#######################
#EVALUATE DATA METRICS#
#######################

#Metric 1. Tau (time of max - time of meal)
#Metric 2. dGN (max glucose - meal glucose)
#Metric 3-6. Pf1, f2, pf2, f2 based on FFT - Not sure how to do this
#Metric 7. dCGM/dt
#Metric 8. d^2(CGM)/dt^2
#Put into a (P + Q) x 8 matrix
#noMealMetrics = []
#mealMetrics = []
combinedMetrics = []
labels = []

for i in mealDataArray:
    features = []

    #1. Find Tau
    maxIndex = i.index(max(i[5:29]))
    tau = maxIndex - 6
    features.append(tau)

    #2. Find dGN
    dGN = (max(i[5:29]) - i[5]) / i[5]
    features.append(dGN)

    
    #3-6. Find pf1, f2, pf2, f2
    n = len(i)
    fhat = np.fft.fft(i, n)
    PSD = (fhat * np.conj(fhat)).real
    #PSD[0] = 0

    pf1 = 0
    f1 = 0
    pf2 = 0
    f2 = 0
    
    for j in range(7, len(PSD) - 1): #Skip first 30 minutes and j = 6
        if PSD[j] > PSD[j-1] and PSD[j] > PSD[j+1]:
            if pf1 == 0:
                pf1 = PSD[j]
                f1 = j - 6
            elif pf2 == 0:
                pf2 = PSD[j]
                f2 = j - 6
                break
    #print(pf1, f1, pf2, f2)
    features.append(pf1)
    features.append(f2)
    features.append(pf2)
    features.append(f2)
    

    #7. dCGM - did not divide by time since delta t is always 5 minutes
    dCGM = (i[7] - i[6])/i[6]
    features.append(dCGM)

    #8. dCGM2 (second derivative)
    dCGM2 = (i[8] - i[6])/i[6]
    features.append(dCGM2)
'''

###################
## MLPClassifier ##
###################
#print(labels)
X_train, X_test, Y_train, Y_test = train_test_split(combinedMetrics, labels, test_size=0.2, random_state=42)

##clf = MLPClassifier(hidden_layer_sizes = (6,5), random_state=5, verbose=True, learning_rate_init=0.01)

##clf.fit(X_train,Y_train)
##YPred = clf.predict(X_test)
#print('YPred shape: ', YPred.shape)
#print(accuracy_score(Y_test,YPred)) #removed accuracy_score metric


#Used to create a test.csv
#df = pd.DataFrame(noMealDataArray)
#df.to_csv('test.csv', header=False, index=False)
'''

###################
## CALCULATE SSE ##
###################
def SSE(dataArray, labels, centers):

    sse = [0 for x in range(len(centers))]

    for i in range(len(dataArray)):
        clusterIndex = labels[i]
        sse[clusterIndex] += (dataArray[i][0] - centers[clusterIndex][0])**2 + (dataArray[i][1] - centers[clusterIndex][1])**2
        
    return sse

pca = PCA(n_components=2).fit(mealDataArray)
mealPCA = pca.transform(mealDataArray)
#print('mealPCA: ', mealPCA)

kmeans = KMeans(n_clusters = numBins, random_state=0).fit(mealPCA)
kmeansLabels = kmeans.labels_
sseArray = SSE(mealPCA, kmeans.labels_, kmeans.cluster_centers_)
kmeansSSE = np.sum(sseArray)


#DBSCAN makes terrible clusters, resort to bisecting k-means
#print('mealDataArray: ', mealDataArray)
#clustering = DBSCAN(eps=100, min_samples=1).fit(mealPCA)
#print('clustering: ', clustering.labels_)
#print('clustering max: ', max(clustering.labels_))

##Bisecting K-means
#bikmeans = KMeans(n_clusters = 2, random_state=0).fit(mealDataArray)


#with open('model_pk1', 'wb') as f:
#    pickle.dump(gbc, f)


############################
## BISECT KMEANS FUNCTION ##
############################

def bisect_KMeans(dataArray, labels, centers, bins):
    sse = SSE(dataArray, labels, centers)
    indexMaxSSE = np.argmax(sse)
    maxIndex = len(sse)

    ##sort dataArray and labels based on labels
    arrlinds = labels.argsort()
    sorted_dataArray = dataArray[arrlinds[::-1]]
    sorted_labels = labels[arrlinds[::-1]]
    sorted_bins = bins[arrlinds[::-1]]

    #print(np.where(sorted_labels == indexMaxSSE))
    splitIndexStart = np.where(sorted_labels == indexMaxSSE)[0][0]
    splitIndexEnd = np.where(sorted_labels == indexMaxSSE)[0][-1]
    #print('splitIndex: ', splitIndexStart, splitIndexEnd)

    
    arrayToSplit = sorted_dataArray[splitIndexStart:splitIndexEnd]

    kmeansArray = KMeans(n_clusters = 2, random_state=0).fit(arrayToSplit)

    ##Replace 0s in label array with prev label
    ##Replace 1s in label array with next label (max(label) + 1)
    kmeansLabels = kmeansArray.labels_
    for i in range(len(kmeansLabels)):
        if kmeansLabels[i] == 0:
            kmeansLabels[i] = indexMaxSSE
        else:
            kmeansLabels[i] = maxIndex
    
    ##Replace old label array section with new labels
    newLabels = np.concatenate([sorted_labels[:splitIndexStart],kmeansLabels,sorted_labels[splitIndexEnd:]])
    kmeansCenters = kmeansArray.cluster_centers_
    newCenters = np.concatenate([centers, [kmeansCenters[1]]])
    newCenters[indexMaxSSE] = kmeansCenters[0]

    return sorted_dataArray, newLabels, newCenters, sorted_bins


#Bisect k-means 6 times (to get 7 clusters)
bisectArray = mealPCA
bisectBins = np.array(carbBin) #ground truth array for bisect kmeans
kmeans_initial = KMeans(n_clusters = 2, random_state=0).fit(mealPCA) ##initial split
bisectLabels = kmeans_initial.labels_
bisectCenters = kmeans_initial.cluster_centers_
for _ in range(5): ##split 5 more times
    bisectArray, bisectLabels, bisectCenters, bisectBins = bisect_KMeans(bisectArray, bisectLabels, bisectCenters, bisectBins)
    
sseArray = SSE(bisectArray, bisectLabels, bisectCenters)
bisectSSE = np.sum(sseArray)
print('bisectBins: ', bisectBins)
print('bisectLabels: ', bisectLabels)

#######################
## CALCULATE ENTROPY ##
#######################

##Create array of probabilities for bisect K-means
numLabels = numBins
arrayOfProbabilities = []
numItems = [0 for x in range(numLabels)]
for i in range(numLabels):

    arrayOfLabel = []

    indexOfItemsInLabel = np.where(bisectLabels == i)
    numItems[i] = len(indexOfItemsInLabel[0])

    itemCount = [0 for x in range(numLabels)]
    for j in indexOfItemsInLabel[0]:
        index = bisectBins[j]
        itemCount[int(index)] += 1
    
    #print('itemCount: ', itemCount)
    
    arrayOfLabel = np.divide(itemCount, numItems[i])
    #print('arrayOfLabel: ', arrayOfLabel)
    arrayOfProbabilities.append(arrayOfLabel)

arrayOfProbabilities = np.array(arrayOfProbabilities)
print(numItems)

##Calculate Entropy
entropyArray = [0 for x in range(numBins)]
purityArray = [0 for x in range(numBins)]
for i in range(numBins):

    purityArray[i] = np.max(arrayOfProbabilities[i])

    for j in arrayOfProbabilities[i]:
        if j != 0.0:
            entropyArray[i] -= j*np.log(j)

bisectEntropy = np.sum(entropyArray)/len(entropyArray)
bisectPurity = np.sum(purityArray)/len(purityArray)

##Create array of probabilities for K Means
arrayOfProbabilities = []
numItems = [0 for x in range(numLabels)]
for i in range(numLabels):

    arrayOfLabel = []

    indexOfItemsInLabel = np.where(kmeansLabels == i)
    numItems[i] = len(indexOfItemsInLabel[0])

    itemCount = [0 for x in range(numLabels)]
    for j in indexOfItemsInLabel[0]:
        index = carbBin[j]
        itemCount[int(index)] += 1
    
    #print('itemCount: ', itemCount)
    
    arrayOfLabel = np.divide(itemCount, numItems[i])
    #print('arrayOfLabel: ', arrayOfLabel)
    arrayOfProbabilities.append(arrayOfLabel)

arrayOfProbabilities = np.array(arrayOfProbabilities)
print(numItems)

##Calculate Entropy
entropyArray = [0 for x in range(numBins)]
purityArray = [0 for x in range(numBins)]
for i in range(numBins):

    purityArray[i] = np.max(arrayOfProbabilities[i])

    for j in arrayOfProbabilities[i]:
        if j != 0.0:
            entropyArray[i] -= j*np.log(j)

kmeansEntropy = np.sum(entropyArray)/len(entropyArray)
kmeansPurity = np.sum(purityArray)/len(purityArray)

with open('Result.csv', 'w') as f:
    csvWriter = csv.writer(f)
    csvWriter.writerow([kmeansSSE, bisectSSE, kmeansEntropy, bisectEntropy, kmeansPurity, bisectPurity])