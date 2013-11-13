import dialog_manager as DM
import q_classify as QC
from nltk.tag import pos_tag
import utility as ut
import nltk
import trysearch as ts
import AlchemyAPI as AP

Dialog_Manager = DM.dialog_manager()
q_class = QC.q_classification()

var = 1
while var:
	
	print '\n'
	string = raw_input(" Enter the question: ")
	
	if string in ["end","End","exit","Exit"]:
		var = 0
	else:
		temp = [ (a,b) for (a,b) in pos_tag(nltk.tokenize.word_tokenize(ut.clean(string)))]
		temp1 = dict()
		temp2 = ''
		for (a,b) in temp:
			if a == 'i':
				a = 'i'
			elif b == 'RB' or b == 'VB':
				temp2 += 'action '
				temp1['action'] = a
			elif b == 'VBP' or b == 'NN':
				temp2 += 'object '
				temp1['object'] = a
			else:
				temp2 += a + ' '
		result_string = Dialog_Manager.get_reply(temp2)
		res_class = q_class.classify([string])
	
		result_string = result_string.replace('action', temp1.get('action',''))
		result_string = result_string.replace('object', temp1.get('object',''))
		print ('Question: '+string+'\t Class : '+str(res_class))
		print ('Question: '+string+'\t Answer: '+result_string)
		
		print ('\n Possible result :\n')
		
		cnt = 0
		web_res = ts.search_google(string)
		text_web = dict()
		max_val_index = 1
		max_val = 0
		for i in web_res:
			obj = AP.AlchemyAPI()
			t = obj.URLGetText(i)
			cnt += 1
			start = '<text>'
			end = '</text>'
			t = (t.split(start))[1].split(end)[0]
			text_web[cnt] = t
			words = nltk.tokenize.word_tokenize(t)
			fdist = nltk.FreqDist(words)
			if len(words) >0:
				temp_val = 1.0 * (fdist[temp1.get('object',0)] + fdist[temp1.get('action',0)]) / len(words)
			if temp_val > max_val:
				max_val_index = cnt
				max_val = temp_val
				max_url = i
		
		#Display final result
		print text_web[max_val_index],'\n'
		print 'Result found at URL: ', i 
