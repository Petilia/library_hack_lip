from multilingual_clip import Config_MCLIP
import transformers
import torch
import clip
# import open_clip

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


def get_text_encode_model(model_name='M-CLIP/XLM-Roberta-Large-Vit-L-14'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultilingualCLIP.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    # model.to(device)
    # tokenizer.to(device)
    return model, tokenizer

def get_image_encode_model(model_name="ViT-L/14"):
    device = "cuda:1" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load(model_name, device=device)
    if model_name == 'M-CLIP/XLM-Roberta-Large-Vit-B-16Plus':
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-16-plus-240', pretrained="laion400m_e32")
    return model, preprocess 