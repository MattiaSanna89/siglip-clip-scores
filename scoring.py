import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from transformers import AutoModel, AutoProcessor, CLIPModel, CLIPProcessor

class MultimodalSimilarity:
    """A class for computing text-image similarity using SigLIP and CLIP models."""
    
    def __init__(self,
        siglip_ckpt = "google/siglip2-giant-opt-patch16-384",
        clip_ckpt = "openai/clip-vit-base-patch32"
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load SigLIP model and processor
        self.siglip_ckpt = siglip_ckpt
        self.siglip_model = AutoModel.from_pretrained(self.siglip_ckpt).to(self.device).eval()
        self.siglip_processor = AutoProcessor.from_pretrained(self.siglip_ckpt)
        self.siglip_max_text_tokens = self.siglip_model.config.text_config.max_position_embeddings

        # Load CLIP model and processor
        self.clip_ckpt = clip_ckpt
        self.clip_model = CLIPModel.from_pretrained(self.clip_ckpt).to(self.device)
        self.clip_processor = CLIPProcessor.from_pretrained(self.clip_ckpt)
        self.clip_max_text_tokens = self.clip_model.config.text_config.max_position_embeddings

        # Define image transformations
        self.transform = transforms.Compose([
            transforms.PILToTensor()
        ])
    
    @staticmethod
    def normalize_feature(embedding):
        """L2 normalizes an embedding tensor."""
        return embedding / embedding.norm(p=2, dim=-1, keepdim=True)
    
    @staticmethod
    def cosine_similarity(text_embeddings, image_embeddings):
        """Computes cosine similarity."""
        return torch.matmul(text_embeddings, image_embeddings.T.to(text_embeddings.device))

    def get_logits(self, text_embeddings, image_embeddings):
        """Computes logits using cosine similarity."""
        logits_per_text = self.cosine_similarity(text_embeddings, image_embeddings)
        
        logit_scale = self.siglip_model.logit_scale.exp().to(text_embeddings.device)
        logit_bias = self.siglip_model.logit_bias.to(text_embeddings.device)

        logits_per_text = logits_per_text * logit_scale + logit_bias
        if logits_per_text.dim() == 1:
            logits_per_text = logits_per_text.unsqueeze(0)
        return logits_per_text, logits_per_text.T

    def get_loss(self, logits_per_text):
        """Computes contrastive loss for multimodal learning."""
        eye = torch.eye(logits_per_text.size(0), device=logits_per_text.device)
        m1_diag1 = -torch.ones_like(logits_per_text) + 2 * eye
        loglik = F.logsigmoid(m1_diag1 * logits_per_text)
        return -torch.sum(loglik, dim=-1).mean()

    def _get_predictions(self, text_embeddings, image_embeddings, return_loss=False):
        """Computes logits, loss, and embeddings."""
        logits_per_text, logits_per_image = self.get_logits(text_embeddings, image_embeddings)
        loss = None
        if return_loss:
            loss = self.get_loss(logits_per_text)

        return {
            "loss": loss,
            "logits_per_image": logits_per_image,
            "logits_per_text": logits_per_text,
            "text_embeddings": text_embeddings,
            "image_embeddings": image_embeddings
        }

    def chunk_text_tokens(self, texts, model_type, overlap):
        # ✅ Select the correct tokenizer & model settings
        if model_type == "siglip":
            processor = self.siglip_processor
            tokenizer = processor.tokenizer
            max_tokens = self.siglip_max_text_tokens
        elif model_type == "clip":
            processor = self.clip_processor
            tokenizer = processor.tokenizer
            max_tokens = self.clip_max_text_tokens
        else:
            raise ValueError("Invalid model_type. Choose 'siglip' or 'clip'.")
        
        unproc_texts = []
        for text in texts:
            # ✅ Tokenize the text without truncation
            tokenized_text = tokenizer(text)['input_ids']
            if len(tokenized_text) <= max_tokens:
                unproc_texts.append(text)  # No chunking needed, keep full text
            else:
                print("here")
                # ✅ Use `chunk_text_tokens()` to split text into chunks
                text_chunks = [
                    tokenized_text[i : i + max_tokens - 2]  
                    for i in range(0, len(tokenized_text), max_tokens -2 - overlap)  # Sliding window
                ]

                # ✅ Convert tokenized chunks back to text
                unproc_texts += [tokenizer.decode(chunk, skip_special_tokens=True) for chunk in text_chunks]

        return unproc_texts

    def encode_inputs(self, texts, images, model_type="siglip", overlap=10):  
        
        if not isinstance(texts, list):
                texts = [texts]
        if not isinstance(images, list):
            images=[images]      

        chunks = self.chunk_text_tokens(texts, model_type, overlap)

        if model_type == "siglip":
            text_inputs = self.siglip_processor(
                                text=chunks,
                                padding="max_length", max_length=64,
                                return_tensors="pt"
                        ).to(self.device)
            
            image_inputs = self.siglip_processor(
                                images=images, 
                                return_tensors="pt"
                        ).to(self.device)
        
        elif model_type == "clip":
            text_inputs = self.clip_processor(
                                text=chunks, 
                                return_tensors="pt",
                                padding=True
                        ).to(self.device)
            
            images_torch = torch.stack(
                [
                    self.transform(img).to(self.device) for img in images
                ]
            )
            image_inputs = self.clip_processor(
                                images=images_torch, 
                                return_tensors="pt", 
                                padding=True
                    ).to(self.device)

        return (text_inputs, image_inputs)


    def predictions(self, texts, images, return_loss = False):
        """Computes text-image predictions using SigLIP."""
        text_inputs, image_inputs = self.encode_inputs(texts, images, model_type="siglip")

        with torch.no_grad():
            image_embeds = self.siglip_model.get_image_features(**image_inputs)
            text_embeds = self.siglip_model.get_text_features(**text_inputs)

        pooled_text_embeds = self.normalize_feature(text_embeds.mean(dim=0))
        image_embeds = self.normalize_feature(image_embeds)
        text_embeds = self.normalize_feature(text_embeds)

        return {
            "pooling_score": self._get_predictions(pooled_text_embeds, image_embeds, return_loss),
            "sequence_score": self._get_predictions(text_embeds, image_embeds, return_loss)
        }

    def siglip_proba(self, texts, images):
        """Computes text-image probabilities using SigLIP."""
        outputs = self.predictions(texts, images)

        for output_type in ["pooling_score", "sequence_score"]:
            logits_per_image = outputs[output_type]["logits_per_image"]
            outputs[output_type]["probabilities"] = torch.sigmoid(logits_per_image)

        return outputs

    def clip_similarity(self, texts, images):
        """Computes text-image similarity using CLIP."""
        text_inputs, image_inputs = self.encode_inputs(texts, images, model_type="clip")

        with torch.no_grad():
            image_clip_emb = self.clip_model.get_image_features(image_inputs["pixel_values"])
            text_clip_emb = self.clip_model.get_text_features(text_inputs["input_ids"], text_inputs["attention_mask"])

        pooled_text_embeds = self.normalize_feature(text_clip_emb.mean(dim=0))
        image_clip_emb = self.normalize_feature(image_clip_emb)
        text_clip_emb = self.normalize_feature(text_clip_emb)

        pooling_sim = self.cosine_similarity(pooled_text_embeds, image_clip_emb) * 100
        seq_sim = self.cosine_similarity(text_clip_emb, image_clip_emb) * 100

        if pooling_sim.dim() == 1:
            pooling_sim = pooling_sim.unsqueeze(0)
        
        if seq_sim.dim() == 1:
            seq_sim = seq_sim.unsqueeze(0)

        return {
            "pooling_score": pooling_sim.T,
            "sequence_score": seq_sim.T
        }
