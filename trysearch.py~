#!/usr/bin/python3
import json
import urllib

def search_google(searchfor):
	query = urllib.urlencode({'q': searchfor})
	url = 'http://ajax.googleapis.com/ajax/services/search/web?v=1.0&rsz=8&%s' % query
	search_response = urllib.urlopen(url)
	search_results = search_response.read().decode("utf8")
	results = json.loads(search_results)
	data = results['responseData']

	hits = data['results']
	
	res = list()

	for h in hits: 
		res.append(h['url'])
	
	return res
