from flask import Flask, request, jsonify
import torch
import os
from kogpt2_transformers import get_kogpt2_tokenizer
from pytorch_lightning import LightningModule, Trainer
from model.kogpt2 import DialogKoGPT2

root_path = '.'
checkpoint_path = f"{root_path}/checkpoint"
save_ckpt_path = f"{checkpoint_path}/kogpt2-wellnesee-auto-regressive.pth"

app = Flask(__name__)

tokenizer = get_kogpt2_tokenizer()

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
            sample_output = self.forward(input_ids)

        answer = self.tokenizer.decode(sample_output[0].tolist()[len(tokenized_indexs) + 1:], skip_special_tokens=True)
        second_dot_index = answer.find('.', answer.find('.') + 1)
        if second_dot_index != -1:
            answer = answer[:second_dot_index + 1]

        return answer


@app.route('/process', methods=['POST'])
def process_data():
    request_data = request.json
    question = request_data.get('question', '')
    answer = dialog_model.inference(question)
    response_data = {
        "answer": answer
    }

    return response_data

if __name__ == '__main__':
    dialog_model = DialogKoGPT2Wrapper(os.path.abspath(save_ckpt_path), tokenizer)
    dialog_model.load_model()
    app.run()
