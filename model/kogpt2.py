import torch.nn as nn
import torch
from pytorch_lightning import LightningModule, Trainer
from kogpt2_transformers import get_kogpt2_model
import time

class DialogKoGPT2(nn.Module):
  def __init__(self):
    super(DialogKoGPT2, self).__init__()
    self.kogpt2 = get_kogpt2_model()

  def generate(self,
               input_ids,
               do_sample=True,
               max_length= 60,
               top_p=0.92,
               top_k=50,
               temperature= 0.6,
               no_repeat_ngram_size =None,
               num_return_sequences=3,
               early_stopping=False,
               ):
    return self.kogpt2.generate(input_ids,
               do_sample=do_sample,
               max_length=max_length,
               top_p = top_p,
               top_k=top_k,
               temperature=temperature,
               no_repeat_ngram_size= no_repeat_ngram_size,
               num_return_sequences=num_return_sequences,
               early_stopping = early_stopping,
              )

  def forward(self, input, labels = None):
    if labels is not None:
      outputs = self.kogpt2(input, labels=labels)
    else:
      outputs = self.kogpt2(input)

    return outputs

class DialogKoGPT2Wrapper(LightningModule):
    def __init__(self, checkpoint_path, tokenizer):
        super(DialogKoGPT2Wrapper, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.tokenizer = tokenizer
        self.model = None
        self.device_type = "cuda" if torch.cuda.is_available() else "cpu"


    def forward(self, input_ids):
        return self.model.generate(input_ids=input_ids, max_length=50)

    def load_model(self):
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
            self.model = DialogKoGPT2().to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
        except Exception as e:
            print(f"Failed to load the model: {e}")
            self.model = None

    def inference(self, question):
        if self.model is None:
            return "네, 듣고있으니 더 말씀해주세요."

        tokenized_indexs = self.tokenizer.encode(question)
        input_ids = torch.tensor(
            [self.tokenizer.bos_token_id, ] + tokenized_indexs + [self.tokenizer.eos_token_id]).unsqueeze(0).to(
            self.device)


        with torch.no_grad():
            start_time = time.time()  # 시작 시간 기록
            sample_output = self.forward(input_ids)
            end_time = time.time()  # 종료 시간 기록
            elapsed_time = end_time - start_time  # 수행 시간 계산
            print(f"inference 시간: {elapsed_time:.4f} seconds")  # 수행 시간 출력

        answer = self.tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:], skip_special_tokens=True)
        second_dot_index = answer.find('.', answer.find('.') + 1)
        if second_dot_index != -1:
            answer = answer[:second_dot_index + 1]

        return answer