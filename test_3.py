from typing import List

from zhipuai import ZhipuAI

from Configs import key

client = ZhipuAI(api_key=key)  # 请填写您自己的APIKey
# 用于保存聊天上下文
history: List = [
    {"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"}]

while True:
    user_input = input("请输入文字，按回车键确认：")
    # 检查用户是否想要退出
    if user_input.lower() == 'exit':
        print("程序退出。")
        break
    # 拼接问题
    history.append({"role": "user", "content": user_input})
    response = client.chat.completions.create(
        model="glm-4",
        messages=history,
        stream=True,
    )
    result = ""
    for chunk in response:
        word = chunk.choices[0].delta.content
        print(word, end='')
        result += word
    # 拼接回答
    history.append({"role": "assistant", "content": result}),
    print("")
