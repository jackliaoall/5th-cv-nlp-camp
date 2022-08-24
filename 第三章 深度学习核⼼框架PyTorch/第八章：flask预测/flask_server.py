import io
import json
import flask
import torch
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
#from torchvision import transforms as T
from torchvision import transforms, models, datasets
from torch.autograd import Variable

# 初始化Flask app
app = flask.Flask(__name__)
model = None
use_gpu = False

# 加载模型进来
def load_model():
    """Load the pre-trained model, you can use your model just as easily.
    """
    global model
    #这里我们直接加载官方工具包里提供的训练好的模型（代码会自动下载）括号内参数为是否下载模型对应的配置信息
    model = models.resnet18()
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_ftrs, 102))  # 类别数自己根据自己任务来

    #print(model)
    checkpoint = torch.load('best.pth')
    model.load_state_dict(checkpoint['state_dict'])
    #将模型指定为测试格式
    model.eval()
    #是否使用gpu

    if use_gpu:
        model.cuda()

# 数据预处理
def prepare_image(image, target_size):
    """Do image preprocessing before prediction on any data.

    :param image:       original image
    :param target_size: target image size
    :return:
                        preprocessed image
    """
    #针对不同模型，image的格式不同，但需要统一至RGB格式
    if image.mode != 'RGB':
        image = image.convert("RGB")

    # Resize the input image and preprocess it.(按照所使用的模型将输入图片的尺寸修改，并转为tensor)
    image = transforms.Resize(target_size)(image)
    image = transforms.ToTensor()(image)

    # Convert to Torch.Tensor and normalize. mean与std   （RGB三通道）这里的参数和数据集中是对应的，训练过程中一致
    image = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)

    # Add batch_size axis.增加一个维度，用于按batch测试   本次这里一次测试一张
    image = image[None]
    if use_gpu:
        image = image.cuda()
    return Variable(image, volatile=True) #不需要求导

# 开启服务   这里的predict只是一个名字，可自定义
@app.route("/predict", methods=["POST"])
def predict():
    # Initialize the data dictionary that will be returned from the view.
    #做一个标志，刚开始无图像传入时为false，传入图像时为true
    data = {"success": False}

    # 如果收到请求
    if flask.request.method == 'POST':
        #判断是否为图像
        if flask.request.files.get("image"):
            # Read the image in PIL format
            # 将收到的图像进行读取
            image = flask.request.files["image"].read()
            image = Image.open(io.BytesIO(image)) #二进制数据

            # 利用上面的预处理函数将读入的图像进行预处理
            image = prepare_image(image, target_size=(64, 64))

            preds = F.softmax(model(image), dim=1)
            results = torch.topk(preds.cpu().data, k=3, dim=1)
            results = (results[0].cpu().numpy(), results[1].cpu().numpy())

            #将data字典增加一个key,value,其中value为list格式
            data['predictions'] = list()

            # Loop over the results and add them to the list of returned predictions
            for prob, label in zip(results[0][0], results[1][0]):
                #label_name = idx2label[str(label)]
                r = {"label": str(label), "probability": float(prob)}
                #将预测结果添加至data字典
                data['predictions'].append(r)

            # Indicate that the request was a success.
            data["success"] = True
    # 将最终结果以json格式文件传出
    return flask.jsonify(data)

"""
test_json = {
                "status_code": 200,
                "success": {
                            "message": "image uploaded",
                            "code": 200
                        },
                "video":{
                    "video_name":opt['source'].split('/')[-1],
                    "video_path":opt['source'],
                    "description":"1",
                    "length": str(hour)+','+str(minute)+','+str(round(second,4)),
                    "model_object_completed":model_flag
                    }
                    "status_txt": "OK"
                    }
                    response = requests.post(
                        'http://xxx.xxx.xxx.xxx:8090/api/ObjectToKafka/',,
                        data={'json': str(test_json)})
"""

if __name__ == '__main__':
    print("Loading PyTorch model and Flask starting server ...")
    print("Please wait until server has fully started")
    #先加载模型
    load_model()
    #再开启服务
    app.run(port='5012')
