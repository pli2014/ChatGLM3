# python -m spacy download zh_core_web_sm
import spacy # 导包
# 加载模型，并排除掉不需要的components
nlp = spacy.load("zh_core_web_sm", exclude=("tagger", "parser", "senter", "attribute_ruler", "ner"))
# 对句子进行处理
doc = nlp("自然语言处理是计算机科学领域与人工智能领域中的一个重要方向。它研究能实现人与计算机之间用自然语言进行有效通信的各种理论和方法。")
# for循环获取每一个token与它对应的向量
for token in doc:
	# 这里为了方便展示，只截取5位，但实际该模型将中文词编码成了96维的向量
    print(token.text, token.tensor[:5]) 