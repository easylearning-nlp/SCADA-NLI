# 地点属性映射
'''
Key: place
Value: id in SCADA
'''
place_dict = {
    '主要' : 2,
    '外面' : 2,
    '客厅' : 3,
    '主卧' : 4,
    '卧室' : 4,
    '卫生间' : 5,
    '厨房' : 6
}

# 查询对象属性映射
'''
Key: Object
Value: Channel ID
'''
object_dict = {
    '多少度' : 1,
    '几度' : 1,
    '温度' : 1,
    '亮度' : 14,
    '湿度' : 2,
    '甲醛' : 3,
    '二氧化碳' :4,
    '空调' : 5,
    '电饭煲' : 6,
    '台灯' : 7,
    '吊灯' : 8,
    '灯' : 9,
    '煤气' : 10,
    '热水器' : 11,
    '窗帘' : 12,
    '电视机' : 13,
    '扫地机器人' : 15,
}


# 控制动作映射
control_dict = {
    '告诉': 0,
    '查看': 0,
    '打开': 1,
    '关闭': 2,
    '开启': 1,
    '关掉': 2,
    '调高': 3,
    '调低': 4,
    '修改': 5
}


# 指令分类
action_dict = {
    'query' : 1,
    'control' : 2
}

ip = "http://192.168.11.151/Scada/View.aspx?viewID="

if __name__ == '__main__':
    print(place_dict.get('卧室'))