import random
import glob
import shutil
import os

# inPath = 'dataset/img/*'
# outPath = 'dataset/img'

def continusRandomSample(inPath, outPath, sampleRate=0.05, continusNum=7):
    image = glob.glob(inPath)
    patients = {}

    if not os.path.exists(outPath):
        os.mkdir(outPath)

    for i in image:
        patient = i.split('.')[1].split('_')  
        if not patients.get(patient[0]):
            patients[patient[0]] = []
        patients[patient[0]].append(int(patient[2]))

    for patient_num, patient_list in patients.items():
        list.sort(patient_list)
        list_temp = list(range(len(patient_list)))
        start = continusNum // 2
        end = len(list_temp) - ( continusNum // 2 ) - 1
        sample_num = int(len(list_temp) * sampleRate) 
        samples = random.sample(list_temp[start: end], sample_num) 
        for s in samples:
            for i in range(continusNum):
                source_filename = ('3Dircadb1.%d_image_%d.png' % (int(patient_num),patient_list[s - start + i]))
                target_filename = ('3Dircadb1.%d_image_%d_%d.png' % (int(patient_num),patient_list[s],i))
                shutil.copy(os.path.join(inPath,source_filename), os.path.join(outPath,target_filename))