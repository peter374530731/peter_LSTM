import pymysql,string

# 打开数据库连接
db = pymysql.connect("172.16.225.21", "root", "SunLand2@", "sscp_py", port=3306, charset='utf8')
with open('output/pred_res_label', 'r', encoding='UTF-8') as f1:
    file_label = f1.readlines()

with open('input/train_data_ID', 'r', encoding='UTF-8') as f2:
    file_ID = f2.readlines()
# 使用 cursor() 方法创建一个游标对象 cursor
cursor = db.cursor()
if len(file_ID)==len(file_label):
    for i in range(24906):
        label = file_label[i].replace("\n", "")
        #print(label)
        Label_ID = int(file_ID[i])
        #print(Label_ID)
        if i%100 ==0:
            print(i)
        #cursor.execute('SET NAMES UTF8')
        # 使用 execute()  方法执行 SQL 查询
        cursor.execute("UPDATE ai_corpus_result_04_0604 SET cluster_result='%s' WHERE consult_id='%d';"% (label, Label_ID))
        #cursor.execute("UPDATE ai_corpus_result_04_0604 SET cluster_result='%s',tag='%s' WHERE consult_id='%d';" % (
        #label, label, Label_ID))
        # 提交到数据库执行
        db.commit()
else:
    print("长度不一致")
# 关闭数据库连接
db.close()