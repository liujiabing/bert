#!/usr/bin/env python
#_*_coding:utf-8_*_

import sys
reload(sys)
sys.setdefaultencoding("utf-8")

import json
import tokenization

lenlist = []
tokenizer = tokenization.FullTokenizer(vocab_file='chinese_L-12_H-768_A-12/vocab.txt')
for line in sys.stdin.readlines():
    t = tokenizer.tokenize(line.strip())
    #print json.dumps(t, ensure_ascii=False)
    lenlist.append(len(t))

print max(lenlist), min(lenlist), sum(lenlist), sum(lenlist)/float(len(lenlist))
larger = [1 if _>256 else 0 for _ in lenlist]
print sum(larger) / float(len(lenlist))
