import os
import random
import shutil
from itertools import islice



outputFolderPath ="Dataset/SplitData"
inputFolderPath="Dataset/all"
splitRatio={"train":0.7,"val":0.2,"test":0.1}
classes=["fake","real"]
try:
    shutil.rmtree(outputFolderPath)

except OSError as e:
    os.mkdir(outputFolderPath)

os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

listNames=os.listdir(inputFolderPath)

uniqueNames=[]
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames=list(set(uniqueNames))



random.shuffle(uniqueNames)


lenData=len(uniqueNames)
lenTrain=int(lenData*splitRatio['train'])
lenVal=int(lenData*splitRatio['val'])
lenTest=int(lenData*splitRatio['test'])


if lenData!=lenTrain+lenTest+lenVal:
    remaining=lenData-(lenTrain+lenTest+lenVal)
    lenTrain+=remaining



lengthToSplit=[lenTrain,lenVal,lenTest]
Input=iter(uniqueNames)
Output=[list(islice(Input,elem))for elem in lengthToSplit]
print(f"Total Images:{lenData} Split: {len(Output[0])} {len(Output[1])} {len(Output[2])}")


sequence=['train','val','test']
for i,out in enumerate(Output):
    for filename in out:
        shutil.copy(f'{inputFolderPath}/{filename}.jpg',f'{outputFolderPath}/{sequence[i]}/images/{filename}.jpg')
        shutil.copy(f'{inputFolderPath}/{filename}.txt',f'{outputFolderPath}/{sequence[i]}/labels/{filename}.txt')

print("Split process Completed")

dataYaml= f'path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc: {len(classes)}\n\
names: {classes}'



f = open(f"{outputFolderPath}/data.yaml", 'a')
f.write(dataYaml)
f.close()

print("Data.yaml file Created...")