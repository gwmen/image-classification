import numpy as np
import pandas as pd
import re
import os

home = 'papers-with-abstracts'
full_path = f"E:/20260130/FGCV-git/{home}"
files = [os.path.join(full_path, o) for o in os.listdir(full_path)]
df_read = pd.concat([pd.read_parquet(o, engine="pyarrow") for o in files])
print()


def extract_containing_all_words(_read):
    _read = _read.values
    results = []
    for ds in _read:
        # try:
        #     # tt = ds[4] if ds[4] else ''
        #     # ab = ds[5] if ds[5] else ''
        #     # tt = ds[4]
        #     # ab = ds[5]
        #     # text = tt + ab
        #     # text = text.lower()
        # except:
        #     print()
        cc = []
        tt = ds[4] if ds[4] else ''
        ab = ds[5] if ds[5] else ''
        text = tt + ab
        text = text.lower()
        # 定义要查找的单词列表
        words = ["fine", "grained", "image", "classification"]
        # words = ['CUB-200-2011']
        # 按句子分割文本
        sentences = re.split(r'[.!?]+', text)
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            # 检查句子是否包含所有关键词
            if all(re.search(rf'\b{word}\b', sentence, re.IGNORECASE) for word in words):
                # results.append(sentence)
                cc.append(sentence)
        if len(cc) > 0:
            results.append(ds)

    return results


if __name__ == '__main__':
    oo = extract_containing_all_words(df_read)
    ooo = np.concatenate([o.reshape(1, -1) for o in oo])
    head = df_read.head()
    df = pd.DataFrame(
        ooo,
        columns=df_read.columns.values,  # 指定列名
        index=list(range(len(ooo)))  # 指定行索引
    )
    df.to_csv('res.csv')
    print(df)
