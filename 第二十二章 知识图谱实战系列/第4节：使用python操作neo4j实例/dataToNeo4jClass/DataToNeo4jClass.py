# -*- coding: utf-8 -*-
from py2neo import Node, Graph, Relationship,NodeMatcher


class DataToNeo4j(object):
    """将excel中数据存入neo4j"""

    def __init__(self):
        """建立连接"""
        link = Graph("http://localhost:7474", username="neo4j", password="tangyudiadid0")
        self.graph = link
        #self.graph = NodeMatcher(link)
        # 定义label
        self.buy = 'buy'
        self.sell = 'sell'
        self.graph.delete_all()
        self.matcher = NodeMatcher(link)
        
        """
        node3 = Node('animal' , name = 'cat')
        node4 = Node('animal' , name = 'dog')  
        node2 = Node('Person' , name = 'Alice')
        node1 = Node('Person' , name = 'Bob')  
        r1 = Relationship(node2 , 'know' , node1)    
        r2 = Relationship(node1 , 'know' , node3) 
        r3 = Relationship(node2 , 'has' , node3) 
        r4 = Relationship(node4 , 'has' , node2)    
        self.graph.create(node1)
        self.graph.create(node2)
        self.graph.create(node3)
        self.graph.create(node4)
        self.graph.create(r1)
        self.graph.create(r2)
        self.graph.create(r3)
        self.graph.create(r4)
        """


    def create_node(self, node_buy_key,node_sell_key):
        """建立节点"""
        for name in node_buy_key:
            buy_node = Node(self.buy, name=name)
            self.graph.create(buy_node)
        for name in node_sell_key:
            sell_node = Node(self.sell, name=name)
            self.graph.create(sell_node)
            
        

    def create_relation(self, df_data):
        """建立联系"""      
        m = 0
        for m in range(0, len(df_data)):
            try:    
                print(list(self.matcher.match(self.buy).where("_.name=" + "'" + df_data['buy'][m] + "'")))
                print(list(self.matcher.match(self.sell).where("_.name=" + "'" + df_data['sell'][m] + "'")))
                rel = Relationship(self.matcher.match(self.buy).where("_.name=" + "'" + df_data['buy'][m] + "'").first(),
                                   df_data['money'][m], self.matcher.match(self.sell).where("_.name=" + "'" + df_data['sell'][m] + "'").first())

                self.graph.create(rel)
            except AttributeError as e:
                print(e, m)
            