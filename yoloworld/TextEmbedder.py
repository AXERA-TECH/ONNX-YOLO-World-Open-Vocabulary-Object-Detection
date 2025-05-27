import clip
import torch


class TextEmbedder(torch.nn.Module):
    def __init__(self, model_name="ViT-B/32", device="cpu"):
        super().__init__()
        self.device = device
        self.clip_model, _ = clip.load(model_name, device=self.device)

    def __call__(self, text_token):
        return self.forward(text_token)
    
    def forward(self, text_token):
        txt_feats = [self.clip_model.encode_text(token).detach() for token in text_token.split(1)]
        txt_feats = torch.cat(txt_feats, dim=0)
        txt_feats /= txt_feats.norm(dim=1, keepdim=True)
        txt_feats = txt_feats.unsqueeze(0)
        return txt_feats
    
    def tokenize(self, text):
        if not isinstance(text, list):
            text = [text]
        return clip.tokenize(text).to(self.device)

    def embed_text(self, text):
        if not isinstance(text, list):
            text = [text]

        text_token = clip.tokenize(text).to(self.device)
        txt_feats = [self.clip_model.encode_text(token).detach() for token in text_token.split(1)]
        txt_feats = torch.cat(txt_feats, dim=0)
        txt_feats /= txt_feats.norm(dim=1, keepdim=True)
        txt_feats = txt_feats.unsqueeze(0)
        return txt_feats


if __name__ == "__main__":
    import numpy as np

    text_embedder = TextEmbedder()
    text = ["cat", "dog"]
    txt_feats = text_embedder.embed_text(text).cpu().numpy()

    np.savez("../data/text_embeddings.npz", txt_feats=txt_feats, text=np.array(text))
