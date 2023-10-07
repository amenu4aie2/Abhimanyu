from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer
import torch
from PIL import Image

class generateCaption:
    def __init__(self) -> None:
        self.modelAddress = "CaptionGenerator_and_FaceRecognition/myModel/snapshots/dc68f91c06a1ba6f15268e5b9c13ae7a7c514084"
        
        self.model = VisionEncoderDecoderModel.from_pretrained(self.modelAddress)
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.modelAddress)
        self.tokenizer = AutoTokenizer.from_pretrained(self.modelAddress)

        self.device = torch.device("cpu")
        self.model.to(self.device)

        self.max_length = 16
        self.num_beams = 4

        self.gen_kwargs = {"max_length": self.max_length, "num_beams": self.num_beams}

        # ----

        self.dangerWords = []



    def generateCaption(self, imagePath):

        image = Image.open(imagePath)
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        pixelValue = self.feature_extractor(images=image, return_tensors="pt").pixel_values.to(self.device)

        outputId = self.model.generate(pixelValue, **self.gen_kwargs)

        pred = self.tokenizer.decode(outputId[0], skip_special_tokens=True)
        preds = pred.strip()

        return preds

    def checkForDangerWords(self, caption):
        for word in self.dangerWords:
            if word in caption:
                return True
        return False
    

if __name__ == "__main__":
    generator = generateCaption()
    print(generator.generateCaption("CaptionGenerator_and_FaceRecognition/testImages/test1.jpg"))
    