from zhipuai import ZhipuAI

from Configs import key


class LlmClient:
    client: object = None
    history: list = [
        {"role": "system",
         "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的回答,问什么就回答什么即可，不用添加过多的内容，如果答案不在经核实的资料中或无法从经核实的资料中得出，请回答:我无法回答您的问题。"}]

    def __init__(self):
        super().__init__()
        self.client = ZhipuAI(api_key=key)

    def query(self, prompt, user_input):
        user_input = "根据历史会话和新已知：[" + prompt + "]。回答以下问题：" + user_input
        self.history.append({"role": "user", "content": user_input})
        print("finally ask::", user_input)
        response = self.client.chat.completions.create(
            model="glm-4",
            messages=self.history,
            stream=True,
        )
        result = ""
        print("result: ", end='')
        for chunk in response:
            word = chunk.choices[0].delta.content
            print(word, end='')
            result += word
        # 拼接回答
        self.history.append({"role": "assistant", "content": result}),
        print("")
