from urllib.parse import unquote
import re
from bs4 import BeautifulSoup
import urllib
import requests
from urllib.request import Request, urlopen, HTTPError
import json

def detect_portal(link):
	workday = re.search("workday", link)
	if(workday):
		return 0

	greenhouse = re.search("greenhouse", link)
	if(greenhouse):
		return 1

	jobslever = re.search("lever", link)
	if(jobslever):
		return 2

	return -1

def extract_key(elem, key):
	if isinstance(elem, dict):
		if key in elem:
			return elem[key]
		for k in elem:
			item = extract_key(elem[k], key)
			if item is not None:
				return item
	elif isinstance(elem, list):
		for k in elem:
			item = extract_key(k, key)
			if item is not None:
				return item

	return None

def workday_job(link):
	res = {}

	is_login = re.search("login", link)
	is_apply = re.search("apply", link)

	if(is_login):
		chunks = link.split("login?redirect=")
		part1 = chunks[0].split(".com")[0] + ".com"
		part2 = chunks[1]

		link = part1+part2
	
	if(is_apply):
		link = re.sub('apply.*', '', link)

	req = Request(link)
	req.add_header("Accept", "application/json,application/xml")
	raw_page = urlopen(req).read().decode()
	page_dic = json.loads(raw_page)

	if(page_dic):
		more_data = extract_key(page_dic, 'structuredDataAttributes')
		json_data = json.loads(more_data['data'])
		res["job_desc"] = extract_key(page_dic, 'description')
		res["date_posted"] = json_data["datePosted"]
		res["req_id"] = json_data["identifier"]["value"]
		res["location"] = json_data["jobLocation"]["address"]["addressLocality"]
		res["job_title"] = json_data["identifier"]["name"]

	return res

def greenhouse_job(link):
	res = {}

	page = requests.get(link)
	soup = BeautifulSoup(page.content,'html.parser')

	header = soup.find("div", {"id":"header"})

	title = header.find("h1").text.strip()
	location = header.find("div", {"class":"location"}).text.strip()
	divs = soup.find("div",{"id":"content"})

	texts = divs.findAll("p")
	description = ""

	for p in texts:
		description += p.text

	res["job_title"] = title
	res["job_desc"] = description
	res["date_posted"] = ""
	res["req_id"] = ""
	res["location"] = location

	return res

def jobslever_job(link):
	res = {}

	is_apply = re.search("apply", link)
	
	if(is_apply):
		link = re.sub('apply.*', '', link)

	page = requests.get(link)
	soup = BeautifulSoup(page.content,'html.parser')

	title_div = soup.find("div", {"class":"posting-headline"})
	
	title = title_div.find("h2").text
	location = title_div.find("div", {"class":"sort-by-time posting-category medium-category-label"}).text

	description = soup.find("div", {"class":"section-wrapper page-full-width"}).text

	res["job_title"] = title
	res["job_desc"] = description
	res["date_posted"] = ""
	res["req_id"] = ""
	res["location"] = location

	return res

def main_method(link):
	url = unquote(link)
	res = detect_portal(url)

	if(res!=-1):
		response = {"message":"Data found"}
		temp = {}

		if(res==0):
			temp = workday_job(url)
		elif(res==1):
			temp = greenhouse_job(url)
		elif(res==2):
			temp = jobslever_job(url)
		response.update(temp)
	else:
		response = {"message":"Data not found"}

	return response