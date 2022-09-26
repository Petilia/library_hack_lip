import torch
import ruclip

class RUCLIPPredictor():
    def __init__(self, device='cuda:0'):
        self.device = device
        self.predictor = self.get_predictor()

    def get_predictor(self, name_ruclip = "ruclip-vit-large-patch14-336"):
        clip, processor = ruclip.load(name_ruclip, device=self.device)
        predictor = ruclip.Predictor(clip, processor, self.device, bs=1)
        return predictor

    def inference(self, text, image):
        # image = Image.open(sample_image_path)
        with torch.no_grad():
            image_features = self.predictor.get_image_latents([image]).cpu().detach().numpy()
            text_features = self.predictor.get_text_latents([text]).cpu().detach().numpy()

        return text_features, image_features

    def inference_text(self, text):
        with torch.no_grad():
            text_features = self.predictor.get_text_latents([text])
        return text_features
    
    def inference_image(self, image):
        with torch.no_grad():
            image_features = self.predictor.get_image_latents([image])
        return image_features

