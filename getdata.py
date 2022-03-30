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
import os

farm_name = '白音查干'
farm_code = '20094'     # 风场编号
wtgs_model = '双馈'       # 机型
wtgs_num = 25           # 风机数


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
    if wtgs_model == '半直驱':
        sql = "SELECT real_time, grWindSpeed, grWindDirction, grOutdoorTemperature, grAirPressure, " \
              "grAirDensity, grGridActivePower, giWindTurbineOperationMode, " \
              "gbTurbinePowerLimited, giWindTurbineYawMode, {} as wtid " \
              "from t{}_all t " \
              "where month(real_time) in (10, 11, 12) " \
              "and DATE_FORMAT(real_time,'%i') % 15 = 0 " \
              "group by DATE_FORMAT(real_time,'%Y-%m-%d %H:%i');".format(wtgs, wtgs)      # 定义SQL语句，每隔15分钟取一条
    elif wtgs_model == '双馈':
        sql = "SELECT real_time, iWindSpeed, iVaneDiiection, wTemp_Operating, '' as grAirPressure, '' as grAirDensity, " \
              "iGenPower, WT_Runcode, iPowerLimit_Flag, iYPLevel, {} as wtid " \
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

    def combine_20094(self):
        """
            合并20094
        :return:
        """
        ori_path = './data/20094/scada'
        dst_path = './data/20094/scada/20094_power_2020.csv'

        csv_list = []
        for (root, dirs, files) in os.walk(ori_path):
            for filename in files:
                if filename.endswith('.xls') or filename.endswith('.xlsx'):
                    csv_list.append(os.path.join(root, filename))

        if len(csv_list) > 0:
            df_list = []
            for filename in csv_list:
                df_tmp = pd.read_excel(filename)
                wtid = int(df_tmp.iloc[1, 0][4:])
                data = df_tmp.iloc[9:, :]
                data.columns = df_tmp.iloc[7, :]
                data_power = data[['记录时间', '发电机功率1秒平均值']]
                data_power.columns = ['real_time', 'power']
                data_power['wtid'] = wtid
                # data.sort_values(by='记录时间', inplace=True)        # 按时间排序
                df_list.append(data_power)
            full_df = pd.concat(df_list)
            full_df.reset_index(inplace=True, drop=True)
            full_df.drop_duplicates(subset=['real_time', 'wtid'], keep='first', inplace=True)  # 去重
            full_df.to_csv(dst_path, index=False)

            # 汇总成全场功率
            full_df.drop(full_df[full_df['power'] == '****'].index, inplace=True)
            farm_power = full_df.groupby(['real_time'])['power'].sum()

            print("合并完毕！")
            return farm_power
        else:
            print("无文件可合并！")
            return pd.DataFrame()

    def combine_v(self, wtgs=''):
        """
            按行拼接，纵向合并多个csv到1个csv
            wtgs：风机编号，决定是否按照风机合并。默认为空，代表合并所有文件。
        """
        print(wtgs)
        csv_list = glob.glob('./data/{}/{}*.csv'.format(farm_code, wtgs))   # wtgs非空时，只合并该台机组的数据
        print("{}共发现{}个csv文件".format(wtgs, len(csv_list)))

        if wtgs == '':
            wtgs = farm_code
        dst_path = './data/{}/{}_2021_v.csv'.format(farm_code, wtgs)      # 保存的目标路径

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

    def combine_h(self, wtgs=''):
        """
            按列拼接，横向合并多个csv到1个csv，比如多个风机点位纵向合并以增加特征，功率相加作为全场功率
        """
        print(wtgs)
        csv_list = glob.glob('./data/{}/{}*.csv'.format(farm_code, wtgs))  # wtgs非空时，只合并该台机组的数据
        print("{}共发现{}个csv文件".format(wtgs, len(csv_list)))

        if wtgs == '':
            wtgs = farm_code
        dst_path = './data/{}/{}_2021_h.csv'.format(farm_code, wtgs)  # 保存的目标路径

        if len(csv_list) > 0:
            df = pd.DataFrame()
            flag = 0        # 标志是否为首个文件
            for filename in sorted(csv_list):
                if flag == 0:
                    df = pd.read_csv(filename)
                    flag = 1
                else:
                    df_tmp = pd.read_csv(filename)
                    # 相同的标签点，后面补充后缀
                    df = pd.merge(df, df_tmp, how='outer', on='real_time', suffixes=['', '_{}'.format(str(flag))])
                    flag += 1
            # 计算全场功率
            all_cols = df.columns       # 所有标签点
            # 所有风机的有功功率字段
            if wtgs_model == "半直驱":
                power_cols = [col for col in all_cols if 'grGridActivePower' in col]
            elif wtgs_model == "双馈":
                power_cols = [col for col in all_cols if 'iGenPower' in col]
            # power_cols = []
            # for col in all_cols:
            #     if 'grGridActivePower' in col:
            #         power_cols.append(col)
            df.set_index('real_time', inplace=True)
            df_new = df[power_cols]
            df_new['farm_power'] = df_new.apply(lambda x: x.sum(), axis=1)      # 所有列求和，得到风场功率

            # 合并特征、风场功率
            df_dst = pd.merge(df, df_new['farm_power'], how='outer', left_index=True, right_index=True).reset_index()
            df_dst.to_csv(dst_path, index=False)
            print("合并完毕！")
            return df_dst
        else:
            print("无文件可合并！")
            return pd.DataFrame()


def get_scada():
    """
        获取scada数据，并进行合并
    :return:
    """
    # 从数据库获取scada数据
    wtgs_list = []
    for i in range(1, wtgs_num + 1):
        wtgs = farm_code + str(i).zfill(3)  # 风机编号
        wtgs_list.append(wtgs)
        data = get_data(wtgs)  # 从数据库获取wtgs的scada数据并保存到本地

    # 合并文件
    # for wtgs in wtgs_list:
    #     cd = CombineData()
    #     data = cd.combine_v(wtgs)     # 按机组合并
    #     data = cd.combine_v()     # 合并所有
    cd = CombineData()
    data = cd.combine_h(wtgs='')
    return data


def get_nwp(nwp_path):
    """
        获取nwp数据，并进行筛选
        筛选规则：
            ① 每个文件取第一个坐标：@id为#1~#100
            ② 同一个气象时刻保留最新一条记录：按照real_time分组，取filename最大的一条记录
    """
    df = pd.read_excel(nwp_path, sheet_name=0)
    print("nwp数据原始有 {} 条记录！".format(len(df)))
    # 筛选①
    df['id_num'] = df.apply(lambda x: int(x['@id'][1:]), axis=1)    # 辅助列
    df_sel = df[(df['id_num'] >= 1) & (df['id_num'] <= 100)]
    print("第一个坐标点有 {} 条记录！".format(len(df_sel)))

    # 筛选②
    df_sel.sort_values(by=['real_time', 'filename'], ascending=[1, 0], inplace=True)    # 排序
    df_res = df_sel.groupby(['real_time']).head(1)      # 分组取第一条
    print("去重后剩余 {} 条记录！".format(len(df_res)))

    return df_res


def nwp_scada():
    """
        nwp数据匹配功率，20094风场
    """
    # 获取20094功率数据
    cd = CombineData()
    data_power = cd.combine_20094()

    # 获取nwp数据
    nwp_path = './data/20094/NWP/all_nwp.xlsx'
    data_nwp = get_nwp(nwp_path)

    # nwp数据匹配功率
    data_nwp.set_index('real_time', inplace=True)
    nwp_power = data_nwp.join(data_power, how='left')
    nwp_power.to_excel('./data/20094/NWP/nwp_power.xlsx', index=True)

    return nwp_power


def main():
    # # 获取scada数据（通用）
    # data = get_scada()

    # 20094风场nwp数据 匹配 功率
    data = nwp_scada()

    return data


if __name__ == '__main__':
    st = datetime.datetime.now()
    main()
    et = datetime.datetime.now()
    dur = (et - st).seconds
    print("\n读数据耗时：{}秒".format(dur))
