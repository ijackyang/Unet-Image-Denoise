# coding:utf-8
import urllib2
import json
import re

Header = {
    #'authorization':'Client-ID d69927c7ea5c770fa2ce9a2f1e3589bd896454f7068f689d8e41a25b54fa6042',
    'accept-version':'v1',
    'Host':'unsplash.com',
    'x-unsplash-client':'web',
    'User-Agent':'Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/57.0.2986.0 Safari/537.36',
    'Referer':'https://unsplash.com/',
    'Connection':'keep-alive',
    'Accept':'*/*'
}
cnt = 1
page_num = 100 #爬取网页翻页次数
page_url = 'https://api.unsplash.com/napi/feeds/home'
for i in xrange(page_num):
    print 'page'+str(i)+':'
    req = urllib2.Request(url=page_url,headers=Header)
    html = urllib2.urlopen(req)
    res = html.read()
    hjson =  json.loads(res)
    next_page = hjson[u'next_page']
    pattern = re.compile('after=(.*)')
    page_bianhao = re.findall(pattern,next_page)[0]
    page_url = 'https://api.unsplash.com/napi/feeds/home?after='+page_bianhao
    print page_url
    photos = hjson[u'photos']
    for each in photos:
        bianhao = each['id']
        pic = urllib2.urlopen('https://unsplash.com/photos/'+bianhao+'/download?force=true').read()
        pic_name = str(cnt)+'.jpg'
        cnt += 1
        print 'download '+pic_name+'...'
        file = open( './pic/'+pic_name, 'wb')
        file.write(pic)
file.close()

