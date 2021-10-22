import torch

def softmax_response(logits):
    return torch.nn.functional.softmax(logits, dim=1)


def attack_confidence_estimation(model, input, label, normalization, proxy=None, epsilon=0.005, epsilon_decay=0.5, max_iterations=15, confidence_score_function=softmax_response, device='cuda'):
    input = input.to(device)
    label = label.to(device)
    model = model.to(device)
    data = normalization(input)
    data.requires_grad = True
    if proxy:
        # Black-box setting, use proxy to calculate the gradients
        proxy = proxy.to(device)
        output = proxy(data)
        proxy.zero_grad()
        with torch.no_grad():
            model_output = model(normalization(input))
    else:
        # White-box setting, use model itself to calculate the gradients
        output = model(data)
        model.zero_grad()
        model_output = output
    init_prediction = model_output.argmax()
    output = confidence_score_function(output)
    # Calculate gradients of model in backward pass
    output[0][init_prediction.item()].backward(retain_graph=True)
    # Collect gradients
    jacobian = data.grad.data
    if init_prediction == label:
        # If the model is correct, we wish to make it less confident of its prediction
        attack_direction = -1
    else:
        # Otherwise, we wish to make it more confident of its misprediction
        attack_direction = 1
    with torch.no_grad():
        for i in range(max_iterations):
            jacobian_sign = jacobian.sign()
            perturbed_image = input + epsilon * jacobian_sign * attack_direction
            perturbed_image = torch.clamp(perturbed_image, 0, 1)
            new_output = model(normalization(perturbed_image))
            if new_output.argmax() == init_prediction:
                # This adversarial example does not change the prediction as required, return it
                return perturbed_image
            else:
                epsilon = epsilon * epsilon_decay
        # The attack has failed; either the epsilon was too large, epsilon_decay too small,
        # or max_iterations was insufficient. Return original input.
        return input