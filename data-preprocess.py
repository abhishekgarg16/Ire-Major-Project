import re


def preprocess(line):
    temp=""
    for i in line:
        if i.isalnum():
            temp+=i.lower()
        elif len(temp) > 0 and temp[-1] != ' ':
            temp+=' '
    return temp


i=0
X=[]
Y=[]

files = ['/home/rama/Desktop/20162029/sem3/IRE/project2/train_test_data/ICHI2016-TrainData.tsv', '/home/rama/Desktop/20162029/sem3/IRE/project2/train_test_data/new_ICHI2016-TestData_label.tsv']
f2 = open('./input-data-merged.txt', 'w')
for file in files:
    with open(file) as f:
        for line in f:
            if i==0:
                i=1
                continue
            line=line.split('\t')
            urls = re.findall('[\(|\[|\{]*http[s]?\:\/\/[A-Za-z0-9\-.:;,!\?\/=+_@#$%^&\*]*[.]*[\)|\]|\}]*', line[2])
            s2 = '++'
            s3 = '**'
            temp2 = "";
            if len(urls)>0:
                urls = ''.join(urls)
                if(s2 in urls):
                    urls = urls.replace(s2,'')
                if(s3 in urls):
                    urls = urls.replace(s3,'')
                #print (urls)
                links = urls.split('http')
                l1 = 'http'
                for link in links:
                    if(link == ''): continue;
                    link = l1+link
                    s2 = link[-10:]
                    s3 = line[2][-11:]
                    if(s2 in s3): 
                        temp = line[2].find(link)
                        line[2] = line[2][:temp]
                    else:
                        temp=""
                        words = re.compile(r'[\:/?=.\-&]+',re.UNICODE).split(link)
                        N = re.search("[0-9]", words[-1])
                        if N:
                            t = words[-1].find(re.findall('[0-9]', words[-1])[-1])
                            words[-1] = words[-1][t+1:]
                        N=re.search("[A-Z]", words[-1])
                        if N:
                            words[-1]=words[-1][N.start():]
                    link = link.replace(words[-1], '')
                    temp1=line[2].find('htt')
                    str1 = line[2][0:temp1]
                    str2 = line[2][temp1 + len(link):]
                    temp2 = (str1+str2);
                    line[2] = temp2
            line1=preprocess(line[1])
            line2=preprocess(line[2])
            f2.write(line[0] + "\t\t" + line1 + " " + line2  + '\n')
    f.close()
f2.close();
