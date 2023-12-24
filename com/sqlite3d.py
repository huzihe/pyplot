'''
Author: hzh huzihe@whu.edu.cn
Date: 2023-12-24 11:06:12
LastEditTime: 2023-12-24 21:01:28
FilePath: /pyplot/com/sqlite3d.py
Descripttion: 
'''
import sqlite3

def SqliteSearch(url,lat,lon):
    # conn = sqlite3.connect('/Users/hzh/Proj/data/smmap-wuhan-wuce1226.db')
    conn = sqlite3.connect(url)
    cursor = conn.cursor()
    # lat = 30.5299097577
    # lon = 114.350192822
    minlat = lat - 0.00036
    maxlat = lat + 0.00036
    minlon = lon - 0.0004
    maxlon = lon + 0.0004
    args = (lon,lon,lat,lat,minlon,maxlon,minlat,maxlat)
    sql = "SELECT longitude,latitude,smmap,(longitude- ?)*(longitude-?)+(latitude-?)*(latitude-?) as distance FROM shadowmap WHERE longitude >= ? AND longitude <= ? AND latitude >= ? AND latitude <= ? order by distance";
    # query = "SELECT longitude,latitude,smmap,(longitude-114.33372280293445)*(longitude-114.33452280293444)+(latitude-30.555712774666695)*(latitude-30.554992774666694) as distance FROM shadowmap WHERE distance<=0.00000032 order by distance";
    cursor.execute(sql,args)
    elstr= ""
    for row in cursor:
        elstr = row[2]
        print(row)
        # return elstr
        break
    conn.close()
    ellist = elstr.split(',')
    ellist = ellist[:-1]
    el = list(map(int, ellist))
    return el

if __name__ == '__main__':
    url = '/Users/hzh/Proj/data/smmap-wuhan-wuce1226.db'
    lat = 30.5299097577
    lon = 114.350192822
    SqliteSearch(url,lat,lon)