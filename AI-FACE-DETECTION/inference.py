from neural_network import FamilyCnn
import torch
from torchvision import transforms
from PIL import Image

def predict_face(imag,idx,weight):
   
    if not isinstance(imag, Image.Image):
        raise TypeError(f"Expected a PIL.Image.Image, but got {type(imag)}")#raise error if image is not PIL.image format
    print(f"[INFO] PIL Image shape before transform: {imag.size}") 

    print(f"[INFO] Type of image received: {type(imag)}")

 
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

 
    idx_to_class = torch.load(idx)

    print(f"dic{idx_to_class}")

    weights = weight
    model = FamilyCnn(num_classes=len(idx_to_class))
    model.load_state_dict(torch.load(weights, map_location=torch.device("cpu")))


    tensor = transform(imag).unsqueeze(0)

  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    tensor = tensor.to(device)


    model.eval() #sets CNN for prediction
    with torch.no_grad(): #gradient descent calculation  will now be False
        output = model(tensor)
        _, predicted = torch.max(output, 1) #predicts the high probabilty face
        result = idx_to_class[predicted.item()]
        print(f"[INFO] Prediction index: {predicted.item()}, Label: {result}")
        return result
    print(f"dic{idx_to_class}")
