import pymysql,string

# 打开数据库连接
db = pymysql.connect("172.16.225.21", "root", "SunLand2@", "sscp_py", port=3306,charset='utf8')

# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()

#cursor.execute('SET NAMES UTF8')
# 使用 execute()  方法执行 SQL 查询
cursor.execute("SELECT consult_id FROM `ai_corpus_result_04_0604` WHERE access_time>=UNIX_TIMESTAMP('2018-04-01') AND  access_time<UNIX_TIMESTAMP('2018-04-08') AND procrules IS  NULL;")

# 使用 fetchone() 方法获取单条数据.
#data = cursor.fetchone()
res = ""
data = cursor.fetchall()
#print(len(data))
for i in range(len(data)):
    tem = data[i][0]
    #res = res + tem.replace(string.punctuation," ")
    tem1 = tem.replace("，", " ")
    tem2 = tem1.replace(",", " ")
    tem3 = tem2.replace("！", " ")
    tem4 = tem3.replace("？", " ")
    tem5 = tem4.replace("]", " ")
    tem6 = tem5.replace("[", " ")
    tem6 = tem6.replace("\n", " ")
    res = res + tem6 +"\n"

with open('input/train_data_raw', 'w', encoding='UTF-8') as f1:
    f1.write(res)


# 关闭数据库连接
db.close()