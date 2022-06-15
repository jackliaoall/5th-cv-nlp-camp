import json

#咱们用最常用的BIO方法
#B-X 代表实体X的开头， I-X代表实体的结尾  O代表不属于任何类型的
def get_token(input):
    #english = 'abcdefghijklmnopqrstuvwxyz0123456789'
    english = 'abcdefghijklmnopqrstuvwxyz'
    output = []
    buffer = ''
    for s in input:
        if s in english or s in english.upper():
            buffer += s
        else:
            if buffer: output.append(buffer)
            buffer = ''
            output.append(s)
    if buffer: output.append(buffer)
    return output

def json2bio(fpath,output,splitby = 's'):
    with open(fpath) as f:
        lines = f.readlines()
        for line in lines:
            annotations = json.loads(line)
            text = annotations['data'].replace('\n',' ')
            all_words = get_token(text.replace(' ',','))
            all_label = ['O'] * len(all_words)
            for i in annotations['label']:
                b_location = i[0]
                e_location = i[1]
                label = i[2]
                all_label[b_location] = 'B-'+label
                if b_location != e_location:
                    for word in range(b_location+1,e_location):
                        all_label[word] = 'I-'+label
            cur_line = 0
            #写入文件
            toekn_label = zip(all_words,all_label)
            with open(output,'a',encoding='utf-8') as f:
                for tl in toekn_label:
                    f.write(tl[0]+str(' ')+tl[1])
                    f.write('\n')
                    cur_line += 1
                    if cur_line == len(all_words):
                        f.write('\n')#空格间隔不同句子



if __name__ == "__main__":
    filename = 'admin.jsonl'
    output = "train_BIO.txt"
    json2bio(filename,output)