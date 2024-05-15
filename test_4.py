from zhipuai import ZhipuAI

from Configs import key

client = ZhipuAI(api_key=key)  # 请填写您自己的APIKey
response = client.chat.completions.create(
    model="glm-4",  # 填写需要调用的模型名称
    messages=[
        {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
        {"role": "user","content": "已知小头的爸爸叫做大头，大头的爸爸叫做老王。请问小头的爷爷叫什么名字？"},
    ],
    stream=True,
)
for chunk in response:
    print(chunk.choices[0].delta.content,end='')
