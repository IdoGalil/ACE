import torch
import torch.nn.functional as F
import timm
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from ace import attack_confidence_estimation

def attack_example(file_name, true_label, transform, normalization):
    image = Image.open(f'./images/{file_name}.jpg').convert('RGB')
    input = transform(image).unsqueeze(0).cuda()  # transform and add batch dimension
    with torch.no_grad():
        output = model(normalization(input))
    orig_prediction = torch.nn.functional.softmax(output, dim=1).max(1)
    print(f'Ground truth label is {true_label}. The predicted label is {orig_prediction[1].item()} with a confidence of {orig_prediction[0].item()}')
    adversarial_example = attack_confidence_estimation(model=model, input=input, label=torch.tensor(true_label), normalization=normalization)
    with torch.no_grad():
        attacked_prediction = torch.nn.functional.softmax(model(normalization(adversarial_example)), dim=1).max(1)
    print(f'After using ACE, the predicted label is still {attacked_prediction[1].item()} with a confidence of {attacked_prediction[0].item()}')

if __name__ == '__main__':
    model = timm.create_model('efficientnet_b0', pretrained=True).cuda()
    model.eval()
    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)
    normalization = transform.transforms.pop(3)
    # A correct prediction example
    print('=============== A correct prediction example: ===============')
    attack_example(file_name='tank', true_label=847, transform=transform, normalization=normalization)
    # An incorrect prediction example
    print('=============== An incorrect prediction example: ===============')
    attack_example(file_name='binoculars', true_label=447, transform=transform, normalization=normalization)