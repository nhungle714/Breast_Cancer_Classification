import pandas as pd
import numpy as np
import os
import pickle

trainFile = "D:\\CBIS-DDSM\mass_case_description_train_set.csv"

os.chdir(os.path.dirname(trainFile))
trainData = pd.read_csv(os.path.basename(trainFile))
os.chdir("../")

trainData.dropna(inplace=True,how='all')

if os.path.dirname(trainFile) == 'mass':
    trainData['label'] = 2
    trainData.loc[trainData['pathology'] == 'MALIGNANT','label'] = 3
else:
    trainData['label'] = 0
    trainData.loc[trainData['pathology'] == 'MALIGNANT','label'] = 1 

trainData = trainData.loc[:,['patient_id','image view','image file path','cropped image file path','ROI mask file path','label']]
pickle.dump(trainData,open(os.path.basename(trainFile)[:-4]+".pkl","wb"))


mass_val = pickle.load(open("mass_case_description_val_set.pkl","rb"))
mass_train = pickle.load(open("mass_case_description_train_set.pkl","rb"))
mass_test = pickle.load(open("mass_case_description_test_set.pkl","rb"))
calc_val = pickle.load(open("calc_case_description_val_set.pkl","rb"))
calc_train = pickle.load(open("calc_case_description_train_set.pkl","rb"))
calc_test = pickle.load(open("calc_case_description_test_set.pkl","rb"))

train = mass_train.append(calc_train, ignore_index=True)
val = mass_val.append(calc_val, ignore_index=True)
test = mass_test.append(calc_test, ignore_index=True)

Patient_id = train.patient_id.unique()
train_patient_id = np.random.choice(Patient_id,size=int(0.7*len(Patient_id)/0.8),replace=False)
test_patient_id = [patient for patient in Patient_id if patient not in train_patient_id]
train_data = train[train['patient_id'].isin(train_patient_id)]
test_data = train[train['patient_id'].isin(test_patient_id)]

pickle.dump(train,open("Train.pkl","wb"))
pickle.dump(val,open("Val.pkl","wb"))
pickle.dump(test,open("Test.pkl","wb"))
