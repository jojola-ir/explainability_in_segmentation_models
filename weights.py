import numpy as np
import timm
import torch
from models import VisionTransformer

IMG_SIZE = 224

# Helpers
def get_n_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)

def assert_tensors_equal(t1, t2):
    a1, a2 = t1.detach().numpy(), t2.detach().numpy()

    np.testing.assert_allclose(a1, a2)


def load_weights(model_name, save):
    model_name = model_name
    model_official = timm.create_model(model_name, pretrained=True)
    model_official.eval()
    print(type(model_official))

    custom_config = {
            "img_size": IMG_SIZE,
            "in_chans": 3,
            "patch_size": 16,
            "embed_dim": 768,
            "depth": 12,
            "n_heads": 12,
            "qkv_bias": True,
            "mlp_ratio": 4,
    }

    model_custom = VisionTransformer(**custom_config)
    model_custom.eval()


    for (n_o, p_o), (n_c, p_c) in zip(
            model_official.named_parameters(), model_custom.named_parameters()
    ):
        assert p_o.numel() == p_c.numel()
        print(f"{n_o} | {n_c}")

        p_c.data[:] = p_o.data

        assert_tensors_equal(p_c.data, p_o.data)

    inp = torch.rand(1, 3, IMG_SIZE, IMG_SIZE)
    res_c = model_custom(inp)
    res_o = model_official(inp)

    # Asserts
    assert get_n_params(model_custom) == get_n_params(model_official)
    assert_tensors_equal(res_c, res_o)

    # Save custom model
    if save:
        torch.save(model_custom, "models/model.pth")

    return model_custom


if __name__ == "__main__":
    print("Done")