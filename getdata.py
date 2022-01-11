"""
    purpose: 获取风功率预测所需数据
    author: lzz
    date: 20211224
    version: v1
"""

import pymysql
import pandas as pd


def db_conn():
    conn = pymysql.connect(host='192.168.0.36', user='liuzhenzhen', passwd='123456', port=5029, db='db30019_2021', charset='utf8')
    return conn


def get_data():
    try:
        conn = db_conn()        # 连接数据库
        print('成功连接数据库！')
    except:
        print('连接数据库失败！')
        return

    cur = conn.cursor()     # 生成游标对象
    sql = "SELECT real_time, grWindSpeed, grWindDirction, grOutdoorTemperature, grAirPressure, " \
          "grAirDensity, grGridActivePower " \
          "from t30019001_all t " \
          "where real_time >= '2021-01-01 00:00:00' " \
          "and DATE_FORMAT(real_time,'%i') % 15 = 0 " \
          "group by DATE_FORMAT(real_time,'%Y-%m-%d %H:%i');"      # 定义SQL语句，每隔几分钟取一条
    cur.execute(sql)        # 执行SQL语句
    data = cur.fetchall()   # 获得数据
    data = pd.DataFrame(data, columns=['real_time', 'grWindSpeed', 'grWindDirction', 'grOutdoorTemperature',
                                       'grAirPressure', 'grAirDensity', 'grGridActivePower'])
    print("读取{}条数据！".format(len(data)))
    print(data.head())
    data.to_csv('./data/30019001_2021_15.csv', index=False)       # 如果sql数据量太大，可以分批读取并存储，每次修改此文件名即可
    cur.close()             # 关闭游标
    conn.close()            # 关闭数据库连接
    return data


def combine_csv():
    """
        合并多个csv到1个csv
    """
    import glob
    csv_list = glob.glob('*.csv')
    print("共发现{}个csv文件".format(len(csv_list)))
    print("正在合并...")
    for i in csv_list:
        fr = open(i, 'rb').read()
        with open('./data/alldata.csv', 'ab') as f:
            f.write(fr)
        print(i)
    print("合并完毕！")


def main():
    data = get_data()
    # combine_csv()
    return data


if __name__ == '__main__':
    main()
