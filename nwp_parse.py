"""
    nwp数据解析
"""

import pandas as pd
import codecs
import re
import glob
from datetime import datetime, timedelta


def single_file_parse(filepath):
    """
        单个文件解析
    :param filepath:  文件路径，eg: './data/20094/NWP/MYCYH2020010108.WPD'
    :return: dataframe
    """
    print(filepath)
    l = []
    try:
        with open(filepath) as file:
            data = file.readlines()[3:-1]
    except:
        try:
            with open(filepath, encoding='utf-8') as file:
                data = file.readlines()[3:-1]
        except:
            with open(filepath, encoding='gb2312') as file:
                data = file.readlines()[3:-1]

    for item in data:
        item_s = re.split(r' +', item)
        l.append(item_s)

    if len(data) > 0:
        df = pd.DataFrame(data=l[1:], columns=l[0])
        filename = filepath[-19:-4]
        df['filename'] = filename
        df['real_time'] = df.apply(lambda x: filename_to_time(x['filename'], x['@id']), axis=1)
        # df.to_excel('{}.xlsx'.format(filename))
    else:
        df = pd.DataFrame()
    return df


def filename_to_time(filename, id):
    """
        根据文件名和文件里的@id字段，生成nwp数据时间
        规则：每天有2个nwp文件，8点和17点各一个；
        气象文件命名规则：MYCYH20200108.WPD、MYCYH20200117.WPD
            MYCYH：为固定标识。
            MYCYH20200108：为2020年 01月 01日 08时，发送（生成）的该文件。
            MYCYH20200117：为20220年 01月 01日 17时，发送（生成）的该文件。
        文件内容解析：（以MYCYH20200108.WPD为例）
            // 113.20     41.45     113.24     41.45     113.20     41.49     113.24     41.49 分别为四个坐标的经度纬度。
            单文件总数据为400条，每100条为一个经纬度数据，按顺序依次是4个经纬度数据，每条数据间隔为1小时（@id字段）
            time= 20191231_21:00:00 是气象预报的开始时间
            @id为#1~#100代表第1个坐标从开始时间 接下来的100个小时的气象预报数据
            @id为#101~#200代表第2个坐标从开始时间 接下来的100个小时的气象预报数据
            @id为#201~#300代表第3个坐标从开始时间 接下来的100个小时的气象预报数据
            @id为#301~#400代表第4个坐标从开始时间 接下来的100个小时的气象预报数据

    :param filename: 文件名
    :param id: 文件里的@id字段
    :return:
    """
    # filename = 'MYCYH2020010108.WPD'
    # id = '#1'
    filedate_str = filename[5:15]       # 截取年月日时
    filehour = filename[13:15]          # 截取小时
    filedate = datetime.strptime(filedate_str, "%Y%m%d%H")
    id_int = 100 if int(id[1:]) % 100 == 0 else int(id[1:]) % 100
    hour_delta = (id_int - 12) if filehour == '08' else (id_int - 9)
    real_time = filedate + timedelta(hours=hour_delta)      # filename 和 id 对应的气象时间

    return real_time


def main():
    files = glob.glob('./data/20094/NWP/*.WPD')
    df_list = []
    for filepath in files:
        df = single_file_parse(filepath)
        df_list.append(df)
    full_df = pd.concat(df_list)
    full_df.to_excel('./data/20094/NWP/all_nwp.xlsx', index=False)
    print("Success !!!")

    # filepath = './data/20094/NWP\MYCYH2020082208.WPD'
    # df = single_file_parse(filepath)


if __name__ == '__main__':
    main()
