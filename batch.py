#!/usr/bin/env python3
import torch
import PIL.Image
from pathlib import Path
import torchvision.transforms.functional as TVF
from torch.utils.data import Dataset, DataLoader
import types
from transformers import SiglipForImageClassification
from tqdm import tqdm

class FolderDataset(Dataset):
    def __init__(self, folder: Path):
        # Only standard extensions
        self.paths = [p for p in folder.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png")]
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        image = PIL.Image.open(self.paths[idx]).convert("RGB")
        if image.size != (512, 512):
            image = image.resize((512, 512), PIL.Image.Resampling.BICUBIC)
        new_image = PIL.Image.new("RGB", (512, 512), (128, 128, 128))
        new_image.paste(image, (0, 0))
        pixel_values = TVF.pil_to_tensor(new_image) / 255.0
        pixel_values = TVF.normalize(pixel_values, [0.5]*3, [0.5]*3)
        return pixel_values, self.paths[idx]

def collate_fn(batch):
    pixels, paths = zip(*batch)
    pixels = torch.stack(pixels)
    return pixels, paths

def joyquality_forward(self, pixel_values: torch.Tensor):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        embedding = self.vision_model(pixel_values).pooler_output
        return self.classifier(embedding).squeeze(-1)

def main():
    folder = Path(__file__).parent / "input"
    dataset = FolderDataset(folder)
    loader = DataLoader(dataset, batch_size=16, num_workers=8, collate_fn=collate_fn)

    model = SiglipForImageClassification.from_pretrained(
        "fancyfeast/joyquality-siglip2-so400m-512-16-o8eg1n4c", dtype=torch.float32
    )
    model.eval()
    model.forward = types.MethodType(joyquality_forward, model)
    model = model.to("cuda")

    # Optional: enable torch.compile for performance (may fail on Windows without Triton)
    # model = torch.compile(model)

    for pixels, paths in tqdm(loader, desc="Scoring images", dynamic_ncols=True):
        pixels = pixels.to("cuda")
        scores = model(pixels).detach().cpu().tolist()
        for score, img_path in zip(scores, paths):
            txt_path = img_path.with_suffix(".txt")
            txt_path.write_text(str(score))

if __name__ == "__main__":
    main()
