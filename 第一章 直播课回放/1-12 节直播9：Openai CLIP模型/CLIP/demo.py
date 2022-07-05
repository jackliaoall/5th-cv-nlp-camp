from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel  # 挂不挂梯子都试试，看看怎么能下载

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")  # openai/clip-vit-base-patch32
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# url = "http://images.cocodataset.org/val2017/000000039769.jpg"
# image = Image.open(requests.get(url, stream=True).raw)
image = Image.open('benchi.jpg')

#text = ["它属于泰迪", "它属于金毛", '它属于藏獒']
#text = ['golden retriever', 'teddy', 'husky']
text = ['Rolls-Royce', 'Audi', 'Benz']
#text = ['God', 'ghost', 'human']
inputs = processor(text=text, images=image, return_tensors="pt", padding=True)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities

for i in range(len(text)):
    print(text[i], ':', probs[0][i])
