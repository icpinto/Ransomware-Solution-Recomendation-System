#--------------------------------------------------------------submit malware binary
'''import requests

REST_URL = "http://localhost:8090/tasks/create/file"
SAMPLE_FILE = "/path/to/malwr.exe"
HEADERS = {"Authorization": "Bearer Q4vDuDX5Gak1NWOMaVkH6g"}

with open(SAMPLE_FILE, "rb") as sample:
    files = {"file": ("temp_file_name", sample)}
    r = requests.post(REST_URL, headers=HEADERS, files=files)

# Add your code to error checking for r.status_code.

task_id = r.json()["task_id"]


# Add your code for error checking if task_id is None.

def submitMalware(malwarePath):
	import requests

	REST_URL = "http://localhost:8090/tasks/create/file"
	SAMPLE_FILE = malwarePath
	HEADERS = {"Authorization": "Bearer Q4vDuDX5Gak1NWOMaVkH6g"}

	with open(SAMPLE_FILE, "rb") as sample:
	    files = {"file": ("temp_file_name", sample)}
	    r = requests.post(REST_URL, headers=HEADERS, files=files)

	# Add your code to error checking for r.status_code.

	task_id = r.json()["task_id"]
  print(task_id)
	return task_id'''

def submitMalware(malwarePath):
	import subprocess
	import re
	cuckoo_command = subprocess.run(["cuckoo", "submit", malwarePath],stdout=subprocess.PIPE,universal_newlines=True)
	#print(cuckoo_command.stdout)
	regex = re.compile(r'#\d\d\d\d\d')
	id = regex.search(cuckoo_command.stdout)
 
	regex2 = re.compile(r'\d\d\d\d\d')
	task_id = regex2.search(id.group())
	return task_id.group()


#----------------------------------------------------------------get json reprt response


'''import requests

url = 'http://localhost:8090/tasks/report/1'

#use the 'headers' parameter to set the HTTP headers:
response = requests.get(url, headers = {"Authorization": "Bearer Q4vDuDX5Gak1NWOMaVkH6g"})
#jsonResponse = response.json()
#print("Entire JSON response")
#print(jsonResponse)

#print(x.text)

with open('report.json', 'wb') as outf:
    outf.write(response.content)

#the 'demopage.asp' prints all HTTP Headers'''

def getAnalysisreport(task_id):
	import requests
        #print("getanalysisreport")

	url = 'http://localhost:8090/tasks/report/%s' % task_id
	response = requests.get(url, headers = {"Authorization": "Bearer Q4vDuDX5Gak1NWOMaVkH6g"})
	

	#save response into file
	filename = "%s.json" % task_id
	with open(filename, 'wb') as outf:
	    outf.write(response.content)



#----------------------------------------------------------convert into api sequence extracted from report into csv file

def extractAPIseq(reportpath,task_id):
	import subprocess
        #print("extractAPI")
	rc = subprocess.call(["./letbash.sh",reportpath,str(task_id)])


#---------------------------------#############################------------------------claassification




def malwareBinaryClassification(filepath,task_id):
	import pickle 
	import pandas as pd
	import numpy as np

	df = pd.read_csv(filepath)
	seq = df['API seq']
	result={}
	#seq
	new_sample = np.array(seq)
	#print(new_sample)
	with open('Pickle_classification_Model.pkl', 'rb') as file:
	    Pickled_Model = pickle.load(file)

	with open('Pickle_ Crowti_Model.pkl', 'rb') as file:
	    Crowti_Model = pickle.load(file)
	with open('Pickle_ Dinome_Model.pkl', 'rb') as file:
	    Dinome_Model = pickle.load(file)
	with open('Pickle_ Lockyenc_Model.pkl', 'rb') as file:
	    Lockyenc_Model = pickle.load(file)
	with open('Pickle_ Locky_Model.pkl', 'rb') as file:
	    Locky_Model = pickle.load(file)
	with open('Pickle_ Reveton_Model.pkl', 'rb') as file:
	    Reveton_Model = pickle.load(file)
	with open('Pickle_ Sorikrypt_Model.pkl', 'rb') as file:
	    Sorikrypt_Model = pickle.load(file)
	with open('Pickle_ Tescrypt_Model.pkl', 'rb') as file:
	    Tescrypt_Model = pickle.load(file)
	with open('Pickle_ Urausy_Model.pkl', 'rb') as file:
	    Urausy_Model = pickle.load(file)


	y_pred = Pickled_Model.predict(new_sample)[0]
	print(y_pred)
	result["predict_fam"] = y_pred

	cols = ["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy", 'predict_fam']
	pred = Pickled_Model.predict_proba(new_sample)
	b=[]
	row=[]
	prob={}
	list1 = pred.tolist()
	for x in list1:
	    b.append(x)
	print(b)
	row.append([b[0][0],b[0][1],b[0][2],b[0][3],b[0][4],b[0][5],b[0][6],b[0][7],Pickled_Model.predict(new_sample)[0]])
	prob["Dinome"] = float(b[0][0])
	prob["Lockyenc"] =float( b[0][1])
	prob["Crowti"] = float(b[0][2])
	prob["Locky"] = float(b[0][3])
	prob["Reveton"] = float(b[0][4])
	prob["Sorikrypt"] = float(b[0][5])
	prob["Tescrypt"] = float(b[0][6])
	prob["Urausy"] = float(b[0][7])
	result["prob_val"]=prob
	df1 = pd.DataFrame(row, columns=cols)
	df = pd.DataFrame(columns=cols)
	df = df.append(df1, ignore_index = True)
	for index, row in df.iterrows():
	    #print(row['predict_fam'])
	    df2=row[["Dinome","Lockyenc","Crowti","Locky","Reveton","Sorikrypt","Tescrypt","Urausy"]]


	    if(row['predict_fam']==' Sorikrypt'):
	        val=Sorikrypt_Model.predict([df2])[0]

	    elif(row['predict_fam']==' Tescrypt'):
	        val=Tescrypt_Model.predict([df2])[0]

	    elif(row['predict_fam']==' Urausy'):
	        val=Urausy_Model.predict([df2])[0]

	    elif(row['predict_fam']==' Locky'):
	        val=Locky_Model.predict([df2])[0]

	    elif(row['predict_fam']==' Reveton'):
	        val=Reveton_Model.predict([df2])[0]

	    elif(row['predict_fam']==' Crowti'):
	        val=Crowti_Model.predict([df2])[0]

	    elif(row['predict_fam']==' Dinome'):
	        val=Dinome_Model.predict([df2])[0]

	    elif(row['predict_fam']==' Lockyenc'):
	        val=Lockyenc_Model.predict([df2])[0]

	print(val)
	result["novelty_val"]=int(val)
	filename = "result_%s.json" % str(task_id)
	import json
	with open(filename, 'w') as fp:
		json.dump(result, fp)
	print(result)


#--------------------------------------------------------------------------------main

def main(filepath):
	from threading import Timer
	import time
	malwarePath = filepath
	task_id=submitMalware(malwarePath)
  #pause execution until finish cuckoo analysis
	time.sleep(240)
	getAnalysisreport(task_id)
	reportpath = "%s.json" % str(task_id)
	extractAPIseq(reportpath,task_id)
	filepath = "%s.csv" % str(task_id)
	malwareBinaryClassification(filepath,task_id)
      

import argparse

if __name__ == "__main__":
    #main()
	try:
        # set it up
		parser = argparse.ArgumentParser(description='Enter malware location')
		parser.add_argument("--f", type=str, default=1, help="Enter malware location")

		# get it
		args = parser.parse_args()
		filepath = args.f

		# use it
		main(filepath)
	except KeyboardInterrupt:
		print('User has exited the program')
