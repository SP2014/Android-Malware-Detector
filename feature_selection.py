'''
Author: Ashish Katlam
Description: Perform feature selection using LinearSVC
'''
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
import pickle
import sys

if __name__ == "__main__":

	file_name = sys.argv[1]
	op_file = open('final_data.csv','w+')
	ffile = open('final_selected_features.txt','w+')
	header = ""
	dataX = []
	dataY = []
	feature_names = []
	with open(file_name,'r') as fp:
		for i,line in enumerate(fp):
			if i == 0:
				header = line
				feature_names = [x for x in line.strip().split(',')[:-1]]
			else:
				dd = [int(x) for x in line.strip().split(',')]
				dataX.append(dd[:-1])
				dataY.append(dd[-1])


	lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(dataX, dataY)
	with open('feature_scores.txt','w+') as aa:
		for a in sorted(zip(map(lambda x: round(x,4), lsvc.coef_[0]),feature_names), reverse=True):
			aa.write(a[1]+'\t'+str(a[0]) + '\n')

	model = SelectFromModel(lsvc, prefit=True)
	pickle.dump( model, open( "feature_model.p", "wb" ) )
	dataX_new = model.transform(dataX)

	final_header = ""
	hh = header.strip().split(',')
	


	for k in lsvc.coef_:
		for j,val in enumerate(k):
			if val == 0.:
				pass
			else:
				final_header+=hh[j]+','
				ffile.write(hh[j]+' : '+str(val) + '\n')
	final_header+='class'
	op_file.write(final_header+'\n')


	for l,dat in enumerate(dataX_new):
		str_to_write = ""
		for v in dat:
			str_to_write+= str(v) + ','
		
		str_to_write+=str(dataY[l])
		op_file.write(str_to_write+'\n')

	ffile.close()
	op_file.close()



