import time
from selenium import webdriver
import pandas as pd
import pickle

phx_from_sql = pickle.load(open( "phx_db.p", "rb" ) )

# Make sure Chrome opens in headless mode
options = webdriver.ChromeOptions()
options.add_argument('headless')
options.add_argument('window-size=1200x600')
# initialize the driver
driver = webdriver.Chrome('/home/brooks/search_crawler/chromedriver', chrome_options=options)



# Point our browser to the right place and let's 
driver.get('https://duckduckgo.com/');
business_ids = phx_from_sql.index.values
for idx in range(len(phx_from_sql)-1,20000,-1):   
	# Make query term - business + "Phoenix"
	business_id = business_ids[idx]
	print(idx, ':' , business_id)
	query1 = phx_from_sql['name'].loc[business_id] + ', Phoenix'

	search_box = driver.find_element_by_name('q')
	search_box.clear()
	search_box.send_keys(query1)
	search_box.submit();
	html = driver.page_source
	time.sleep(1)
	savestr = './htmlfiles/'+business_id + '.html'

	#Write HTML to file
	file = open(savestr,'w') 
	file.write(html) 
	file.close()

	# Wait a couple seconds before we rinse and repeat
	time.sleep(0.5)