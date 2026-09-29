"""Visual encoders used for HiRE reward computation."""

def create_encoder(name="dino", device="cpu"):
    from .vision import build_similarity_encoder
    return build_similarity_encoder(name, device=device)
