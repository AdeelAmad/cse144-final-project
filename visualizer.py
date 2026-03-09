import torch
from model import Model
from torchviz import make_dot
from torchinfo import summary

def visualize_model(ckpt_path, device='mps'):
    model_obj = Model(device)
    model = model_obj.model
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    
    output = model(dummy_input)
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    dot.render('model_architecture', format='png', cleanup=True)
    print("Model architecture visualization saved as 'model_architecture.png'")


def model_summary(device='mps'):
    model_obj = Model(device)
    model = model_obj.model

    summary(model, input_size=(1, 3, 224, 224))

if __name__ == "__main__":
    ckpt_path = 'checkpoints/best_final_cnn.pt'
    model_summary()
    visualize_model(ckpt_path)
