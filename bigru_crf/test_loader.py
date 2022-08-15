from torchtext.legacy import data, datasets



WORD = data.Field(batch_first=True, include_lengths=True)
TAG = data.Field(batch_first=True, include_lengths=True)
train_set, val_set, test_set = datasets.UDPOS.splits(path='./',
                                                             train='text.bmes',
                                                             validation='text.bmes',
                                                             test='text.bmes',
                                                             fields=(('word', WORD), ('tag', TAG)),
                                                             separator=' ')
       
       
       
       
print("train_set: ", train_set)


       

WORD.build_vocab(train_set.word, val_set.word, test_set.word)
TAG.build_vocab(train_set.tag, val_set.tag, test_set.tag)

char2idx = WORD.vocab.stoi
id2char = WORD.vocab.itos
tag2idx = TAG.vocab.stoi
idx2tag = {index: value for index, value in enumerate(TAG.vocab.itos)}
print(f'char2idx: {char2idx}, tag2idx: {tag2idx}')


train_iter, val_iter, test_iter = data.BucketIterator.splits(
    (train_set, val_set, test_set),
    batch_size=2,
    sort_within_batch=True,
    shuffle=True
)

batch = next(iter(train_iter))
print(batch)






x = batch.word[0]
y = batch.tag[0]
real_length = batch.word[1]

print(x)
print(y)
print(real_length)


