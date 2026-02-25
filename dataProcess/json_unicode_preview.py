'''
预览 JSON 文件中的 Unicode 字符
使用方法：
1. 将你的 JSON 文件路径替换为 `path` 变量中的路径。
2. 运行脚本，它将打印出 JSON 文件中前几条数据，并显示数据的结构信息。
注意：确保你的 JSON 文件使用 UTF-8 编码，以正确处理非 ASCII 字符。
'''

import json


path = "./data/json/test.json"  # 你的这段文件
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)  # 自动把 \uXXXX 还原

# 看前两条，ensure_ascii=False 用于“打印中文”
print(json.dumps(data[:3], ensure_ascii=False, indent=2))

# data 的结构应当是：[[en, zh], [en, zh], ...]
print(type(data), len(data), type(data[0]), len(data[0]))
