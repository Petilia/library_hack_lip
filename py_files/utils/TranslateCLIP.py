import torch
import torch
import open_clip
from transformers import pipeline
from open_clip import tokenizer

class TranslateCLIP():
    def __init__(self, device='cuda:0', translator_device=0):
        self.device = device
        self.translator_device = translator_device
        self.predictor, self.preprocess = self.get_predictor()
        self.translator = self.get_translator()

    def get_predictor(self, model='ViT-H-14', weights='laion2b_s32b_b79k'):
        model, _, preprocess = open_clip.create_model_and_transforms(
            'ViT-H-14', 
            pretrained=weights, 
            device=torch.device(self.device))
        model.eval();     
        return model, preprocess

    def get_translator(self, device=0, model_checkpoint = "Helsinki-NLP/opus-mt-ru-en"):
        translator = pipeline("translation", model=model_checkpoint, device=device)
        return translator

    def inference(self, text, image):
        text_features = self.inference_text(text)
        image_features = self.inference_image(image)
        return text_features, image_features

    def inference_text(self, text):
        with torch.no_grad():
            translated_text = self.translator(text, truncation=True)[0]['translation_text']
            tok = tokenizer.tokenize(translated_text)
            tok = tok.to(self.device)
            text_features = self.predictor.encode_text(tok).cpu().numpy()

        return text_features
    
    def inference_image(self, image):
        image = image.convert("RGB")
        image = self.preprocess(image)
        image = image.to(self.device)
        with torch.no_grad():
            image_features = self.predictor.encode_image(image[None, ...]).cpu().numpy()
        return image_features