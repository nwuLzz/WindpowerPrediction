"""
    purpose: 获取风功率预测所需数据
    author: lzz
    date: 20211224
    version: v1
"""

import pymysql
import pandas as pd
import datetime
import glob


farm_code = '30008'     # 风场编号


def db_conn():
    conn = pymysql.connect(host='192.168.0.36', user='liuzhenzhen', passwd='123456', port=5029, db='db'+farm_code+'_2021', charset='utf8')
    return conn


def get_data(wtgs):
    try:
        conn = db_conn()        # 连接数据库
        print('成功连接数据库！')
    except:
        print('连接数据库失败！')
        return
    print(wtgs)
    cur = conn.cursor()     # 生成游标对象

    # 取多个月份的数据
    sql = "SELECT real_time, grWindSpeed, grWindDirction, grOutdoorTemperature, grAirPressure, " \
          "grAirDensity, grGridActivePower, giWindTurbineOperationMode, " \
          "gbTurbinePowerLimited, giWindTurbineYawMode, {} as wtid " \
          "from t{}_all t " \
          "where month(real_time) in (10, 11, 12) " \
          "and DATE_FORMAT(real_time,'%i') % 15 = 0 " \
          "group by DATE_FORMAT(real_time,'%Y-%m-%d %H:%i');".format(wtgs, wtgs)      # 定义SQL语句，每隔15分钟取一条
    cur.execute(sql)        # 执行SQL语句
    data = cur.fetchall()   # 获得数据
    data = pd.DataFrame(data, columns=['real_time', 'grWindSpeed', 'grWindDirction', 'grOutdoorTemperature',
                                       'grAirPressure', 'grAirDensity', 'grGridActivePower',
                                       'giWindTurbineOperationMode', 'gbTurbinePowerLimited', 'giWindTurbineYawMode', 'wtid'])
    print("读取{}条数据！".format(len(data)))
    data.to_csv('./data/{}/{}_2021_15.csv'.format(farm_code, wtgs), index=False)   # 如果sql数据量太大，可以分批读取并存储，每次修改此文件名即可
    cur.close()             # 关闭游标

    '''
    # 按月读数据
    for month_id in range(10, 13):     # 按月取
        cur = conn.cursor()     # 生成游标对象
        sql = "SELECT real_time, grWindSpeed, grWindDirction, grOutdoorTemperature, grAirPressure, " \
              "grAirDensity, grGridActivePower, giWindTurbineOperationMode, " \
              "gbTurbinePowerLimited, giWindTurbineYawMode, {} as wtid " \
              "from t{}_all t " \
              "where month(real_time) = {} " \
              "and DATE_FORMAT(real_time,'%i') % 15 = 0 " \
              "group by DATE_FORMAT(real_time,'%Y-%m-%d %H:%i');".format(wtgs, wtgs, month_id)      # 定义SQL语句，按月读取，每隔几分钟取一条
        cur.execute(sql)        # 执行SQL语句
        data = cur.fetchall()   # 获得数据
        data = pd.DataFrame(data, columns=['real_time', 'grWindSpeed', 'grWindDirction', 'grOutdoorTemperature',
                                           'grAirPressure', 'grAirDensity', 'grGridActivePower',
                                           'giWindTurbineOperationMode', 'gbTurbinePowerLimited', 'giWindTurbineYawMode', 'wtid'])
        print("读取{}条数据！".format(len(data)))
        data.to_csv('./data/{}/{}_2021_{}_15.csv'.format(farm_code, wtgs, month_id), index=False)   # 如果sql数据量太大，可以分批读取并存储，每次修改此文件名即可
        cur.close()             # 关闭游标
    '''
    conn.close()            # 关闭数据库连接
    return data


class CombineData:
    """
        数据整合，风机级合并成风场级
    """
    def __init__(self):
        pass

    def combine_v(self, wtgs=''):
        """
            纵向合并多个csv到1个csv
            wtgs：风机编号，决定是否按照风机合并。默认为空，代表合并所有文件。
        """
        print(wtgs)
        csv_list = glob.glob('./data/{}/{}*.csv'.format(farm_code, wtgs))   # wtgs非空时，只合并该台机组的数据
        print("机组{}共发现{}个csv文件".format(wtgs, len(csv_list)))

        if wtgs == '':
            wtgs = farm_code
        dst_path = './data/{}/{}_2021_15.csv'.format(farm_code, wtgs)      # 保存的目标路径

        if len(csv_list) > 0:
            df_list = []
            for filename in sorted(csv_list):
                df_tmp = pd.read_csv(filename)
                df_tmp.sort_values(by='real_time', inplace=True)        # 按时间排序
                df_tmp.drop_duplicates(subset=['real_time'], keep='first', inplace=True)        # 去重
                # df_tmp['wtid'] = filename[13:21]
                df_list.append(df_tmp)
            full_df = pd.concat(df_list)
            full_df.to_csv(dst_path, index=False)

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

    def combine_h(self):
        """
            横向合并多个csv到1个csv，比如多个风机点位纵向合并以增加特征，功率相加作为全场功率
        """
        pass


def main():
    wtgs_list = []
    for i in range(10, 11):
        wtgs = farm_code + str(i).zfill(3)    # 风机编号
        wtgs_list.append(wtgs)
        data = get_data(wtgs)       # 从数据库获取wtgs的scada数据并保存到本地

    # # 合并文件
    # for wtgs in wtgs_list:
    #     cd = CombineData()
    #     data = cd.combine_v(wtgs)     # 按机组合并
    #     data = cd.combine_v()     # 合并所有

    return data


if __name__ == '__main__':
    st = datetime.datetime.now()
    main()
    et = datetime.datetime.now()
    dur = (et - st).seconds
    print("\n读scada数据耗时：{}秒".format(dur))
