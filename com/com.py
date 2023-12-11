#!/usr/bin/python
import math

RE = 6378137

def rms(list):
	"""
    @author   : Lizhen
    Version   : 1.0 Created  2019.08.06
    Purpose   : 计算列表的均方根误差
    Input     : list     ——  待计算均方根的列表
    Output    : cal      ——  列表的均方根
	"""
	from math import sqrt
	cal = 0.0
	for each in list:
		cal += each * each
	cal = sqrt(cal / len(list))
	return cal

def std(list):
	"""
    @author   : Lizhen
    Version   : 1.0 Created  2019.08.10
    Purpose   : 计算列表的标准差
    Input     : list     ——  待计算标准差的列表
    Output    : cal      ——  列表的标准差
	"""
	from math import sqrt
	cal = 0.0
	list_mean = mean(list)
	for each in list:
		cal += (each - list_mean) ** 2
	cal = sqrt(cal / len(list))
	return cal

def mean(list):
	"""
    @author   : Lizhen
    Version   : 1.0 Created  2019.08.10
    Purpose   : 计算列表的平均值
    Input     : list     ——  待计算平均值的列表
    Output    : cal      ——  列表的平均值
	"""
	cal = 0.0
	for each in list:
		cal += each
	cal = cal / len(list)
	return cal


def maxabs(list):
	"""
    @author   : Lizhen
    Version   : 1.0 Created  2022.05.7
    Purpose   : 计算列表中绝对值最大的值
    Input     : list     ——  待计算平均值的列表
    Output    : cal      ——  绝对值最大值
    """
	cal = list[0]

	for each in list:
		if abs(each) > cal:
			cal = abs(each)
	return cal

def xyz_blh(x,y,z):
	"""
    @author   : Lizhen
    Version   : 1.0 Created  2019.08.10
    Purpose   : xyz2blh
	"""
	RAD2DEG = 180.0/math.pi
	a = 6378137
	e2 = (1.0/298.257223563) * (2.0- 1.0/298.257223563)
	r2 = x*x + y*y + z*z - z*z
	zz = z
	zk = 0

	while(math.fabs(zz-zk) >= 1e-4):
		zk = zz
		sinp = zz / math.sqrt(r2 + zz * zz)
		v = a / math.sqrt(1.0 - e2 * sinp *sinp)
		zz = z +v * e2 * sinp

	if r2 > 1e-12:
		b = math.atan(zz / math.sqrt(r2))
		l = math.atan2(y, x)
		h = math.sqrt(r2 + zz*zz) - v
	else:
		return 0,0,0
	return b*RAD2DEG, l*RAD2DEG, h

def hav(theta):
    s = math.sin(theta / 2)
    return s * s

def get_distance_hav(lat0, lng0, lat1, lng1):
    """用haversine公式计算球面两点间的距离。"""
    # 经纬度转换成弧度
    lat0 = math.radians(lat0)
    lat1 = math.radians(lat1)
    lng0 = math.radians(lng0)
    lng1 = math.radians(lng1)
 
    dlng = math.fabs(lng0 - lng1)
    dlat = math.fabs(lat0 - lat1)
    h = hav(dlat) + math.cos(lat0) * math.cos(lat1) * hav(dlng)
    distance = 2 * RE * math.asin(math.sqrt(h))
    return distance

def get_deltB(lat0, lng0, lat1, lng1):
	if lat1<lat0:
		return get_distance_hav(lat0, lng1, lat1, lng1)
	else:
		return -get_distance_hav(lat0, lng1, lat1, lng1)
	
def get_deltL(lat0, lng0, lat1, lng1):
	if lng1<lng0:
		return get_distance_hav(lat1, lng0, lat1, lng1)
	else:
		return -get_distance_hav(lat1, lng0, lat1, lng1)