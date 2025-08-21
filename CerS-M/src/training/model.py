import torch
import torch.nn as nn
import timm
# from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
# import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen2_5_vl
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, AutoProcessor
from transformers.modeling_utils import get_parameter_dtype

from torch.utils.data import DataLoader
import sys
sys.path.append("./")
from qwen2_5_model_base.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLPatchMerger
from qwen2_5_model_base.processing_qwen2_5_vl import Qwen2_5_VLProcessor
import pdb


class VisionTransformer(nn.Module):
    def __init__(self, model_name, config, load_pretrained=False):
        super(VisionTransformer, self).__init__()
        self.model = timm.create_model(model_name, img_size=config.img_size, patch_size=16,
                                       init_values=1e-5, num_classes=0, pretrained=False)
        if load_pretrained:
            self.model.load_state_dict(torch.load(config.pretrained_model_path))
        self.merger = Qwen2_5_VLPatchMerger(
            dim=config.out_hidden_size,
            context_dim=config.visual_merge_size,
            spatial_merge_size=config.spatial_merge_size,
        )

    @property
    def dtype(self):
        return get_parameter_dtype(self)

    def forward(self, x, grid_thw=None):
        x = self.model.forward_features(x)[:, 1:, :]
        x = self.merger(x)
        return x


class Qwen2_5_Custom_Vit(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.visual = VisionTransformer("vit_large_patch16_224", config.vision_config)


if __name__ == "__main__":
    # change the visual model in Qwen2_5_VLForConditionalGeneration
    # model = Qwen2_5_VLForConditionalGeneration.from_pretrained("qwen2_5_vl")
    # model.visual_model = VisionTransformer("vit_base_patch16_224", config)
    # tokenizer = Qwen2_5_VLTokenizer.from_pretrained("qwen2_5_vl")
    from data import SupervisedDataset, DataCollatorForSupervisedDataset
    from params import DataArguments
    from transformers import HfArgumentParser

    # test the model
    config = AutoConfig.from_pretrained("./qwen2_5_model_base")
    print(config.architectures)
    visual = VisionTransformer("vit_large_patch16_224", config.vision_config)
    image = torch.randn(1, 3, 224, 224)
    output = visual(image)
    print(output.shape)

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained("./qwen2_5_model_base", torch_dtype=torch.bfloat16)
    processor = Qwen2_5_VLProcessor.from_pretrained("./qwen2_5_model_base", padding_side="right")
    model.visual = visual
    parser = HfArgumentParser(DataArguments)
    data_args = parser.parse_args_into_dataclasses()[0]
    print(data_args)
    dataset = SupervisedDataset(data_path="/data/chenli/PathCap/4_PathCap/data.json", processor=processor,
                                data_args=data_args, model_id="./qwen2_5_model_base")
    data_collator = DataCollatorForSupervisedDataset(pad_token_id=processor.tokenizer.pad_token_id)
    dataloader = DataLoader(dataset, collate_fn=data_collator, batch_size=2)
    # print("dataset_prepared")
    # for batch in dataloader:
    #     out = model(**batch)
    #     break
    labels = dataset[0].pop("labels")
    # print(dataset[0])
    # out = model(**dataset[0])
    # print(out.shape)
    #
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {"type": "image"},
    #             {"type": "text", "text": "What is shown in this image?"},
    #         ],
    #     },
    # ]
    # url = "https://www.ilankelman.org/stopsigns/australia.jpg"
    # image = Image.open(requests.get(url, stream=True).raw)
    #
    # text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # inputs = processor(text=[text], images=[image], vision_infos=[vision_infos])
    #
    #
