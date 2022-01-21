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


def get_data(wtgs):
    try:
        conn = db_conn()        # 连接数据库
        print('成功连接数据库！')
    except:
        print('连接数据库失败！')
        return
    print(wtgs)
    for i in range(3, 13):  # 3~12月
        cur = conn.cursor()     # 生成游标对象
        sql = "SELECT real_time, grWindSpeed, grWindDirction, grOutdoorTemperature, grAirPressure, " \
              "grAirDensity, grGridActivePower, giWindTurbineOperationMode, " \
              "gbTurbinePowerLimited, giWindTurbineYawMode " \
              "from t{}_all t " \
              "where month(real_time) = {} " \
              "and DATE_FORMAT(real_time,'%i') % 15 = 0 " \
              "group by DATE_FORMAT(real_time,'%Y-%m-%d %H:%i');".format(wtgs, i)      # 定义SQL语句，按月读取，每隔几分钟取一条
        cur.execute(sql)        # 执行SQL语句
        data = cur.fetchall()   # 获得数据
        data = pd.DataFrame(data, columns=['real_time', 'grWindSpeed', 'grWindDirction', 'grOutdoorTemperature',
                                           'grAirPressure', 'grAirDensity', 'grGridActivePower',
                                           'giWindTurbineOperationMode', 'gbTurbinePowerLimited', 'giWindTurbineYawMode'])
        print("读取{}条数据！".format(len(data)))
        data.to_csv('./data/'+wtgs+'_2021_'+'{}'.format(i)+'_15.csv', index=False)   # 如果sql数据量太大，可以分批读取并存储，每次修改此文件名即可
        cur.close()             # 关闭游标
    conn.close()            # 关闭数据库连接
    return data


def combine_csv(wtgs):
    """
        合并多个csv到1个csv
    """
    import glob

    csv_list = glob.glob('./data/{}*.csv'.format(wtgs))
    print("机组{}共发现{}个csv文件".format(wtgs, len(csv_list)))
    if len(csv_list) > 0:
        df_list = []
        for filename in sorted(csv_list):
            df_tmp = pd.read_csv(filename)
            df_tmp.sort_values(by='real_time', inplace=True)        # 按时间排序
            df_tmp.drop_duplicates(subset=['real_time'], keep='first', inplace=True)        # 去重
            df_list.append(df_tmp)
        full_df = pd.concat(df_list)
        full_df.to_csv('./data/'+wtgs+'_2021_15.csv', index=False)

        # 以下写法会导致每个文件的表头都保留到最终文件
        # for i in csv_list:
        #     fr = open(i, 'rb').read()
        #     with open('./data/30019001_2021_15.csv', 'ab') as f:
        #         f.write(fr)
        #     print(i)
        print("合并完毕！")
        return full_df
    else:
        print("无文件可合并！")
        return pd.DataFrame()


def combine_csv():
    """
        合并多个csv到1个csv
    """
    import glob

    csv_list = glob.glob('./data/*.csv')
    print("共发现{}个csv文件".format(len(csv_list)))
    if len(csv_list) > 0:
        df_list = []
        for filename in sorted(csv_list):
            df_tmp = pd.read_csv(filename)
            df_tmp.sort_values(by='real_time', inplace=True)        # 按时间排序
            df_tmp.drop_duplicates(subset=['real_time'], keep='first', inplace=True)        # 去重
            df_tmp['wtid'] = filename[7:15]
            df_list.append(df_tmp)
        full_df = pd.concat(df_list)
        print(full_df.head().append(full_df.tail()))
        full_df.to_csv('./data/30019_2021_15.csv', index=False)

        print("合并完毕！")
        return full_df
    else:
        print("无文件可合并！")
        return pd.DataFrame()


def main():
    # wtgs_list = []
    # for i in range(1, 8):
    #     wtgs = str(30019000 + i)
    #     wtgs_list.append(wtgs)
    #
    # # 按机组合并文件
    # for wtgs in wtgs_list:
    #     # data = get_data(wtgs)
    #     data = combine_csv(wtgs)

    # 合并所有文件
    data = combine_csv()
    return data


if __name__ == '__main__':
    main()
