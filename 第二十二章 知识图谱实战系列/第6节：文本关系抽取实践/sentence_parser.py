import os
from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
# pip install pyltp -i https://pypi.tuna.tsinghua.edu.cn/simple 可以先下载好whl文件
#LTP语言平台：http://ltp.ai/index.html
#咱们使用的工具包,pyltp:https://pyltp.readthedocs.io/zh_CN/latest/api.html
#LTP附录：https://ltp.readthedocs.io/zh_CN/latest/appendix.html#id3
#安装方法：https://github.com/HIT-SCIR/pyltp
class LtpParser:
    def __init__(self):
        LTP_DIR = "./ltp_data_v3.4.0"
        self.segmentor = Segmentor()
        self.segmentor.load(os.path.join(LTP_DIR, "cws.model"))

        self.postagger = Postagger()
        self.postagger.load(os.path.join(LTP_DIR, "pos.model"))

        self.parser = Parser()
        self.parser.load(os.path.join(LTP_DIR, "parser.model"))

        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load(os.path.join(LTP_DIR, "ner.model"))

        self.labeller = SementicRoleLabeller()
        self.labeller.load(os.path.join(LTP_DIR, 'pisrl_win.model'))

    '''语义角色标注'''
    def format_labelrole(self, words, postags):
        arcs = self.parser.parse(words, postags)
        roles = self.labeller.label(words, postags, arcs)
        roles_dict = {}
        for role in roles:
            roles_dict[role.index] = {arg.name:[arg.name,arg.range.start, arg.range.end] for arg in role.arguments}
        return roles_dict

    '''句法分析---为句子中的每个词语维护一个保存句法依存儿子节点的字典'''
    def build_parse_child_dict(self, words, postags, arcs):
        child_dict_list = []
        format_parse_list = []
        for index in range(len(words)):
            child_dict = dict()
            for arc_index in range(len(arcs)):
                if arcs[arc_index].head == index+1:   #arcs的索引从1开始 arc. head 表示依存弧的父结点的索引。 ROOT 节点的索引是 0 ，第一个词开始的索引依次为1，2，3，···arc. relation 表示依存弧的关系。
                    if arcs[arc_index].relation in child_dict:
                        child_dict[arcs[arc_index].relation].append(arc_index)#添加
                    else:
                        child_dict[arcs[arc_index].relation] = []#新建
                        child_dict[arcs[arc_index].relation].append(arc_index)
            child_dict_list.append(child_dict)# 每个词对应的依存关系父节点和其关系
        rely_id = [arc.head for arc in arcs]  # 提取依存父节点id
        relation = [arc.relation for arc in arcs]  # 提取依存关系
        heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]  # 匹配依存父节点词语
        for i in range(len(words)):
            a = [relation[i], words[i], i, postags[i], heads[i], rely_id[i]-1, postags[rely_id[i]-1]]
            format_parse_list.append(a)

        return child_dict_list, format_parse_list

    '''parser主函数'''
    def parser_main(self, sentence):
        words = list(self.segmentor.segment(sentence))
        postags = list(self.postagger.postag(words))
        arcs = self.parser.parse(words, postags)
        child_dict_list, format_parse_list = self.build_parse_child_dict(words, postags, arcs)
        roles_dict = self.format_labelrole(words, postags)
        return words, postags, child_dict_list, roles_dict, format_parse_list


if __name__ == '__main__':
    parse = LtpParser()
    #sentence = '我想听一首迪哥的歌'
    sentence = '奥巴马昨晚在白宫发表了演说'
    words, postags, child_dict_list, roles_dict, format_parse_list = parse.parser_main(sentence)
    print(words, len(words))
    print(postags, len(postags))
    print(child_dict_list, len(child_dict_list))
    print(roles_dict)
    print(format_parse_list, len(format_parse_list))