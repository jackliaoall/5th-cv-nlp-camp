# -*- coding: utf-8 -*-
from dataToNeo4jClass.DataToNeo4jClass import DataToNeo4j
import os
import pandas as pd
#pip install py2neo==5.0b1 注意版本，要不对应不了

invoice_data = pd.read_excel('./Invoice_data_Demo.xls', header=0, encoding='utf8')
#print(invoice_data)

#可以先阅读下文档：https://py2neo.org/v4/index.html

def data_extraction():
    """节点数据抽取"""

    # 取出购买方名称到list
    node_buy_key = []
    for i in range(0, len(invoice_data)):
        node_buy_key.append(invoice_data['购买方名称'][i])
    
    node_sell_key = []
    for i in range(0, len(invoice_data)):
        node_sell_key.append(invoice_data['销售方名称'][i])
        
    # 去除重复的发票名称
    node_buy_key = list(set(node_buy_key))
    node_sell_key = list(set(node_sell_key))

    # value抽出作node
    node_list_value = []
    for i in range(0, len(invoice_data)):
        for n in range(1, len(invoice_data.columns)):
            # 取出表头名称invoice_data.columns[i]
            node_list_value.append(invoice_data[invoice_data.columns[n]][i])
    # 去重
    node_list_value = list(set(node_list_value))
    # 将list中浮点及整数类型全部转成string类型
    node_list_value = [str(i) for i in node_list_value]

    return node_buy_key, node_sell_key,node_list_value


def relation_extraction():
    """联系数据抽取"""

    links_dict = {}
    sell_list = []
    money_list = []
    buy_list = []

    for i in range(0, len(invoice_data)):
        money_list.append(invoice_data[invoice_data.columns[19]][i])#金额
        sell_list.append(invoice_data[invoice_data.columns[10]][i])#销售方方名称
        buy_list.append(invoice_data[invoice_data.columns[6]][i])#购买方名称


    # 将数据中int类型全部转成string
    sell_list = [str(i) for i in sell_list]
    buy_list = [str(i) for i in buy_list]
    money_list = [str(i) for i in money_list]

    # 整合数据，将三个list整合成一个dict
    links_dict['buy'] = buy_list
    links_dict['money'] = money_list
    links_dict['sell'] = sell_list
    # 将数据转成DataFrame
    df_data = pd.DataFrame(links_dict)
    print(df_data)
    return df_data

relation_extraction()
create_data = DataToNeo4j()

create_data.create_node(data_extraction()[0], data_extraction()[1])
create_data.create_relation(relation_extraction())
