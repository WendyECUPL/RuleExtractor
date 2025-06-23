import json
# 数据预处理管道
def formatting_prompts_func_sft(sample):
    """构建结构化生成模板"""
    #_instruction = sample["instruction"]
    _instruction = '''从给定的法律条文中抽取规则，并结构化存储。每条规则应包括以下字段：
- rule_id：由规则的出处自动生成，如果规则文本可拆解出多条规则，则用_num 进行编号,例如“《中华人民共和国刑法》第二条_1”。
- subject：规则的主体，指负责执行或管理该规则的机构或个人。
- object：规则的对象，指受规则约束的机构或个人。
- test：规则的触发条件，描述对象需要满足的条件。
- consequence：规则的后果，描述对象在满足 test 条件后可能面临的措施或处罚。
- rule_type：5种规则类型，'义务性规则'、'禁止性规则'、'授权性规则'、'原则性规则'、'鼓励性规则'，如果不在5种之一设为 '其他'”。
- reference：规则的出处，例如具体的法律法规条款,例如"《中华人民共和国刑法(2023修正)》第一编　总则 第一章　刑法的任务、基本原则和适用范围 第二条"。
- depend_on：若该规则依赖于其他规则，则列出依赖规则的 rule_id，否则设为 'null'。
- tag：法条的简单总结性描述，在原文中可能使用【】符合标识，也可能没有，需要根据法条内容总结。
对于所有字段，如果从规则文本中提取不出内容，则填写null。规则的主体和对象不同时为空。
'''
    _input = sample["original"]["text"]
    _output = json.dumps(sample["groundtruth"], ensure_ascii=False, indent=2)
    #example_output = json.dumps(sample["output"], ensure_ascii=False, indent=2)
    
    prompt = f"""请严格按JSON格式从input(法律条文)中提取法律规则:
### input(法律条文)：
{_input}

### Instruction（当前任务）：
{_instruction}

### Response  (按JSON格式生成抽取结果，一个列表):
{_output}
"""
    #print(prompt)
    #target = f"{example_output}\n</json_end>"  # 明确目标输出
    return {"text": prompt, }
    
def formatting_prompts_func_na(sample):
    """构建结构化生成模板"""
    #_instruction = sample["instruction"]
    _instruction = '''从给定的法律条文中抽取规则，并结构化存储。每条规则应包括以下字段：
- rule_id：由规则的出处自动生成，如果规则文本可拆解出多条规则，则用_num 进行编号,例如“《中华人民共和国刑法》第二条_1”。
- subject：规则的主体，指负责执行或管理该规则的机构或个人。
- object：规则的对象，指受规则约束的机构或个人。
- test：规则的触发条件，描述对象需要满足的条件。
- consequence：规则的后果，描述对象在满足 test 条件后可能面临的措施或处罚。
- rule_type：5种规则类型，'义务性规则'、'禁止性规则'、'授权性规则'、'原则性规则'、'鼓励性规则'，如果不在5种之一设为 '其他'”。
- reference：规则的出处，例如具体的法律法规条款,例如"《中华人民共和国刑法(2023修正)》第一编　总则 第一章　刑法的任务、基本原则和适用范围 第二条"。
- depend_on：若该规则依赖于其他规则，则列出依赖规则的 rule_id，否则设为 'null'。
- tag：法条的简单总结性描述，在原文中可能使用【】符合标识，也可能没有，需要根据法条内容总结。
对于所有字段，如果从规则文本中提取不出内容，则填写null。
    '''
    _input = sample["original"]["text"]
    _output = json.dumps(sample["groundtruth"], ensure_ascii=False, indent=2)
    _predict = json.dumps(sample["processed"], ensure_ascii=False, indent=2)
    _label = json.dumps(sample["label"], ensure_ascii=False, indent=2)

    #example_output = json.dumps(sample["output"], ensure_ascii=False, indent=2)
    
    prompt = f"""请严格按JSON格式从法律条文中提取信息，并对已有错误结果进行纠正。

### input (法律条文)：
{_input}

### model_output (模型的抽取结果)：
{_predict}

### Instruction：
你将收到一段法律条文、一组模型的抽取结果（model_output）。请：
1. 阅读条文；
2. 按如下指令进行法律规则抽取：
{_instruction}
3. 对抽取结果（model_output）中的字段进行纠正，如果标记为“正确”，错误则标记为“错误，应该为:”,输出label格式为JSON。

### Response (按JSON格式生成抽取结果，一个rule列表，其中每一个rule为一个字典):
{_label}

"""
    
    #print(prompt)
    #target = f"{example_output}\n</json_end>"  # 明确目标输出
    return {"text": prompt, } 