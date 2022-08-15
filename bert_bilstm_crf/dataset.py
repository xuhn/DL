import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler
from transformers import AutoTokenizer, AutoModelForMaskedLM
import collections
import torch
import utils

class Vocab: 
    """文本词表"""
    # 从dict_path导入字典
    def __init__(self, dict_path, unk=None):
        self.unk_token = unk
        self.idx_to_token = []
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}

        for line in open(dict_path, 'r', encoding='utf-8'):
            token = line.strip('\n')
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # 未知词元的索引为0
        oov_id = self.token_to_idx.get(self.unk_token) if self.unk_token else 0
        return oov_id

class BertDataSet(Dataset):
    __doc__ = """ bert格式dataset  """

    def __init__(self, data_path,
                       label_dic,
                       max_seq_length,
                       tokenizer):
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer
        self.label_dic = label_dic

        self.input_ids = []
        self.input_mask = []
        self.segment_ids = []
        self.label_ids = []
        self._build(data_path, max_seq_length, tokenizer)
    
    def _read(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as fp:
            next(fp)
            features_label_list = []

            for line in fp.readlines():
                words, names = line.strip('\n').split('\t')
                words = words.split('\002')
                names = names.split('\002')
                features_label_list.append((" ".join(words), names))  

        return features_label_list


    def _build(self, data_path, max_seq_length, tokenizer):
        features_label_list = self._read(data_path)
        for cl in features_label_list:
            content, label = cl
            res = self.tokenizer.encode_plus(content,
                                             max_length=max_seq_length, 
                                             padding='max_length')

            #[CLS]和[SEP]对应的标签均是['O']，添加到标签序列中
            label = ['O'] + label + ['O']
            #生成由标签编码构成的序列
            label = [self.label_dic[x] for x in label]

            # pad label            
            label += [utils.IGNORE_LABEL] * (max_seq_length - len(label))
           
            self._store(input_ids=res['input_ids'],
                        input_mask=res['attention_mask'],
                        segment_ids=res['token_type_ids'],
                        label_ids=label,
                        max_seq_length=max_seq_length)
    
    def _store(self, input_ids, input_mask, segment_ids, label_ids, max_seq_length):
        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        
        self.input_ids.append(input_ids)
        self.input_mask.append(input_mask)
        self.segment_ids.append(segment_ids)
        self.label_ids.append(label_ids)
    
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return torch.tensor(self.input_ids[item]), torch.tensor(self.input_mask[item]), torch.tensor(self.segment_ids[item]), torch.tensor(self.label_ids[item])

if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-roberta-wwm-ext")
    dic = Vocab('./tag.txt', 'O')
    ds = BertDataSet('./demo.txt', dic, 50, tokenizer)
    
    utils.format_s(ds[0])