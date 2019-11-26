# -*- coding: utf-8 -*-

import urllib2
import re
import shutil

html = urllib2.urlopen("https://freemidi.org/new").read()
urlmid = re.findall(r'\(<a href=(.*?).mid>', html)
# urlpdf = re.findall(r'\(<a href=(.*?).pdf">',html)

for item in urlmid:
	ff = item.replace("/", "_").replace("-", "_")
	url = "https://freemidi.org/new" + item[1:] + '.mid'

	req = urllib2.urlopen(url)
	with open(ff[1:] + '.mid', 'w') as fp:
		shutil.copy(req, fp)


