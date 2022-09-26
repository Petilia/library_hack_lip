from multilingual_clip import Config_MCLIP
import transformers
import torch
import clip

class MultilingualCLIP(transformers.PreTrainedModel):
    config_class = Config_MCLIP.MCLIPConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config, *args, **kwargs)
        self.transformer = transformers.AutoModel.from_pretrained(config.modelBase, cache_dir=kwargs.get("cache_dir"))
        self.LinearTransformation = torch.nn.Linear(in_features=config.transformerDimensions,
                                                    out_features=config.numDims)

    def forward(self, txt, tokenizer):
        txt_tok = tokenizer(txt, padding=True, return_tensors='pt', truncation=True)
        embs = self.transformer(**txt_tok)[0]
        att = txt_tok['attention_mask']
        embs = (embs * att.unsqueeze(2)).sum(dim=1) / att.sum(dim=1)[:, None]
        return self.LinearTransformation(embs)

    @classmethod
    def _load_state_dict_into_model(cls, model, state_dict, pretrained_model_name_or_path, _fast_init=True):
        model.load_state_dict(state_dict)
        return model, [], [], []


class MLCLIPPredictor():
    def __init__(self, device='cuda:0'):
        self.device = device
        self.text_model, self.text_tokenizer = self.get_text_encode_model()
        self.image_model, self.image_preproc = self.get_image_encode_model()

    def get_text_encode_model(self, model_name='M-CLIP/XLM-Roberta-Large-Vit-L-14'):
        model = MultilingualCLIP.from_pretrained(model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length=512)
        return model, tokenizer

    def get_image_encode_model(self, model_name="ViT-L/14"):
        model, preprocess = clip.load(model_name, device=self.device)
        return model, preprocess 

    def inference(self, text, image):
        # image = Image.open(sample_image_path)
        image = self.image_preproc(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.image_model.encode_image(image).cpu().detach().numpy()

        text_features = self.text_model.forward(text, self.text_tokenizer).cpu().detach().numpy()

        return text_features, image_features
