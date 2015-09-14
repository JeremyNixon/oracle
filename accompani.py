import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from selenium.common.exceptions import ElementNotVisibleException
from selenium.webdriver.common.keys import Keys


visited = {}
queue = []

def get_names(source, visited, level):
    for i in range(len(source)):
        if '"_GCg' in source[i]:
            while 'title' not in source[i]:
                i += 1
                if 'title' in source[i]:
                    first = source[i][7:]
                    if source[i+1][-1] == '"':
                        last = source[i+1][:-1]
                    else:
                        last = source[i+1]
                    name = ' '.join([first, last])
                    if name not in queue:
                        queue.append(name)
                        visited[name] = level

def collect_names(driver, level):
    time.sleep(2)
    page_source = driver.page_source.split()
    get_names(page_source, visited, level)

def bfs(level):


    driver = webdriver.Chrome()
    driver.wait = WebDriverWait(driver, 5)
    driver.get('https://www.google.com/search?q=mark+zuckerberg&ei=O8xoVePlOYWzyQSv24KYBA#q=mark+zuckerberg&stick=H4sIAAAAAAAAAGOovnz8BQMDgwkHsxCnfq6-gYVZSl6lEoKpJZWdbKVfkJpfkJMKpIqK8_OsijNTUssTK4snv3XxDSnam1eqFGX9xK6X50Xr8bsAbYLy7FEAAAA')
    collect_names(driver, 1)
    print "run"

    for i in queue:
        if visited[i] < level:
            try:
                box = driver.wait.until(EC.presence_of_element_located((By.NAME, "q")))
                box.clear()
                box.send_keys(i)
                box.send_keys(Keys.RETURN)
                time.sleep(1)
            except TimeoutException:
                pass

            try:
                button = driver.find_element_by_xpath("//a[contains(., 'People also search for')]")
                button.click()
                if 'Google Search' not in driver.title:
                    driver.back()
                collect_names(driver, visited[i]+1)

            except NoSuchElementException:
                pass
            except ElementNotVisibleException:
                pass
    driver.quit()
    return queue, visited
    

queue, visited = bfs(level = 3)