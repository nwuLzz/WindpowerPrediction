"""
    purpose: 风功率预测
    author: lzz
    date: 20211222
    version: v1
"""
import math


class A:
    def __init__(self, p1, p2=0.8):
        self.p1 = p1
        self.p2 = p2

    def fun_a(self):
        print(self.p1)
        print(self.p2)


def v_to_p(v, r, c_p):
    """
        风速转为功率
    :param v: 风速
    :param r: 扇叶长度
    :param c_p: 风能利用系数，根据贝茨极限，最大为59.3%，目前高性能风机的c_p一般为40%~45%
    :return: 功率
    """
    p = 0.5 * 1.293 * (0.5 * math.pi * math.pow(r, 2)) * math.pow(v, 3) * c_p / 1000        # 单位：kW
    return round(p, 2)


def main():
    # v = 4.35
    # r = 121
    # c_p = 0.3
    # p = v_to_p(v, r, c_p)
    # print("风速：{} m/s    功率：{} kW".format(v, p))
    a = A(1)
    a.fun_a()


if __name__ == '__main__':
    main()
