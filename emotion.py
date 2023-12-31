# 감정 분류 추론 모델 로드

# pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'
# pip install "git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf" #쌍따옴표로 하니까 됨


# PATH = r'C:\Welover\Flask-hanul\kobert_state_ver3.pt'
PATH = r'/home/ubuntu/Flask-hanul/checkpoint/kobert_state_ver3.pt'

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np

from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup
from get_data import preprocess_text

device = torch.device('cpu')
# device = torch.device('cuda')
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1', return_dict=False)
vocab = nlp.vocab.BERTVocab.from_sentencepiece(tokenizer.vocab_file, padding_token='[PAD]')

max_len = 64
batch_size = 64
warmup_ratio = 0.1
num_epochs = 5
max_grad_norm = 1
log_interval = 200
learning_rate = 5e-5

class BERTClassifier(nn.Module):
    def __init__(self,
                bert,
                hidden_size = 768,
                num_classes=8,
                dr_rate=None,
                params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size , num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids = token_ids, token_type_ids = segment_ids.long(), attention_mask = attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        return self.classifier(out)

class BERTSentenceTransform:

    def __init__(self, tokenizer, max_seq_length,vocab, pad=True, pair=True):
        self._tokenizer = tokenizer
        self._max_seq_length = max_seq_length
        self._pad = pad
        self._pair = pair
        self._vocab = vocab

    def __call__(self, line):
        # convert to unicode
        text_a = line[0]
        if self._pair:
            assert len(line) == 2
            text_b = line[1]

        tokens_a = self._tokenizer.tokenize(text_a)
        tokens_b = None

        if self._pair:
            tokens_b = self._tokenizer(text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b,
                                    self._max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self._max_seq_length - 2:
                tokens_a = tokens_a[0:(self._max_seq_length - 2)]

        vocab = self._vocab
        tokens = []
        tokens.append(vocab.cls_token)
        tokens.extend(tokens_a)
        tokens.append(vocab.sep_token)
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens.extend(tokens_b)
            tokens.append(vocab.sep_token)
            segment_ids.extend([1] * (len(tokens) - len(segment_ids)))

        input_ids = self._tokenizer.convert_tokens_to_ids(tokens)

        # The valid length of sentences. Only real  tokens are attended to.
        valid_length = len(input_ids)

        if self._pad:
            # Zero-pad up to the sequence length.
            padding_length = self._max_seq_length - valid_length
            # use padding tokens for the rest
            input_ids.extend([vocab[vocab.padding_token]] * padding_length)
            segment_ids.extend([0] * padding_length)

        return np.array(input_ids, dtype='int32'), np.array(valid_length, dtype='int32'),\
            np.array(segment_ids, dtype='int32')

class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, vocab, max_len, pad, pair):
        transform = BERTSentenceTransform(bert_tokenizer, max_seq_length=max_len,vocab=vocab, pad=pad, pair=pair)
        #transform = nlp.data.BERTSentenceTransform(
        #    tokenizer, max_seq_length=max_len, pad=pad, pair=pair)
        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i], ))

    def __len__(self):
        return (len(self.labels))

# 감정 분류

c_model = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
checkpoint = torch.load(PATH, map_location='cpu')
c_model.load_state_dict(checkpoint["model"], strict=False)
c_model.eval()

def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    c_model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = c_model(token_ids, valid_length, segment_ids)

        emotion_label = ["분노" ,"슬픔", "기쁨", "걱정", "불안감", "중립", "우울감", "공포감"]

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append(0)
            elif np.argmax(logits) == 1:
                test_eval.append(1)
            elif np.argmax(logits) == 2:
                test_eval.append(2)
            elif np.argmax(logits) == 3:
                test_eval.append(3)
            elif np.argmax(logits) == 4:
                test_eval.append(4)
            elif np.argmax(logits) == 5:
                test_eval.append(5)
            elif np.argmax(logits) == 6:
                test_eval.append(6)
            elif np.argmax(logits) == 7:
                test_eval.append(7)

        return test_eval[0]

def predict2(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    c_model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = c_model(token_ids, valid_length, segment_ids)

        emotion_label = ["분노" ,"슬픔", "기쁨", "걱정", "불안감", "중립", "우울감", "공포감"]

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval.append("fear")
            elif np.argmax(logits) == 1:
                test_eval.append("Surprised")
            elif np.argmax(logits) == 2:
                test_eval.append("anger")
            elif np.argmax(logits) == 3:
                test_eval.append("sad")
            elif np.argmax(logits) == 4:
                test_eval.append("neutrality")
            elif np.argmax(logits) == 5:
                test_eval.append("happy")
            elif np.argmax(logits) == 6:
                test_eval.append("Disgust")

        return test_eval[0]
    

def predict3(chat_data):
    # 전체 채팅 메시지를 전처리하여 리스트에 저장
    preprocessed_chat_data = [preprocess_text(chat) for chat in chat_data]

    dataset_another = [(chat, '0') for chat in preprocessed_chat_data]

    another_test = BERTDataset(dataset_another, 0, 1, tokenizer, vocab, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=0)

    c_model.eval()

    # 감정 카운트 딕셔너리 초기화
    emotion_count = {
        "fear": 0,
        "Surprised": 0,
        "anger": 0,
        "sad": 0,
        "neutrality": 0,
        "happy": 0,
        "Disgust": 0
    }

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = c_model(token_ids, valid_length, segment_ids)

        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                emotion_count["fear"] += 1
            elif np.argmax(logits) == 1:
                emotion_count["Surprised"] += 1
            elif np.argmax(logits) == 2:
                emotion_count["anger"] += 1
            elif np.argmax(logits) == 3:
                emotion_count["sad"] += 1
            elif np.argmax(logits) == 4:
                emotion_count["neutrality"] += 1
            elif np.argmax(logits) == 5:
                emotion_count["happy"] += 1
            elif np.argmax(logits) == 6:
                emotion_count["Disgust"] += 1

    # 가장 많이 등장한 감정 찾기
    max_emotion = max(emotion_count, key=emotion_count.get)

    return max_emotion  
