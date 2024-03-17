corpus_post = 'clean_chat_corpus/stc_weibo_train_post'  # 假定的第一个文件路径
corpus_response = 'clean_chat_corpus/stc_weibo_train_response'  # 假定的第二个文件路径


combined_lines = []  # 初始化用于保存配对行的列表
with open(file1, 'r', encoding='utf-8') as f1, open(file2, 'r', encoding='utf-8') as f2:
    for line1, line2 in zip(f1, f2):
        combined_lines.append([line1.strip(), line2.strip()])  # 移除行尾的换行符，并将两行作为列表保存

for line in combined_lines:
    sentences = []
    for value in line:
        sentence = jieba.lcut(cop.sub("", value))
        sentence = sentence[:max_sentence_length] + [eos]
        sentences.append(sentence)
    data.

# 调用函数并打印结果示例
# combined_lines = read_files_line_by_line(corpus_post, corpus_response)
# print(combined_lines[:10])  # 打印前10对行的示例
