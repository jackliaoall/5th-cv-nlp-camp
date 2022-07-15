import medical_re
import json

#medical_re.load_schema("D:\\eclipse-workspace\\huggingface\\CMeKG\\predicate.json")
model4s, model4po = medical_re.load_model()

text = '据报道称，新冠肺炎患者经常会发热、咳嗽，少部分患者会胸闷、乏力，其病因包括: 1.自身免疫系统缺陷\n2.人传人。'  # content是输入的一段文字
res = medical_re.get_triples(text, model4s, model4po)
print(json.dumps(res, ensure_ascii=False, indent=True))