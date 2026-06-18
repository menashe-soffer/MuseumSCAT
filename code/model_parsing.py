import torch
import torch.nn as nn
from PIL import Image


def custom_model_parser(model, max_depth=3):
    print(f"{'DEPTH':<6} | {'MODULE PATH':<60} | {'CLASS TYPE'}")
    print("-" * 110)

    for name, module in model.named_modules():
        # Clean the name string for the base layer
        if name == "":
            name = "root"

        # Count depth by tracking string dots
        depth = len(name.split('.')) if name != "root" else 0

        # Skip layers that are buried deeper than your focus area
        if depth > max_depth:
            continue

        class_name = module.__class__.__name__

        # Add a visual indentation prefix based on depth
        indent = "  " * depth

        # Optional: Extract parameter metrics if it has weights
        param_info = ""
        if hasattr(module, 'in_features') and hasattr(module, 'out_features'):
            param_info = f" ({module.in_features} -> {module.out_features})"

        print(f"{depth:<6} | {indent + name:<60} | {class_name}{param_info}")


import torch


# class ExecutionFlowTracker:
#     def __init__(self):
#         self.flow = []
#         self.hooks = []
#
#     def __enter__(self):
#         return self
#
#     def __exit__(self, exc_type, exc_val, exc_tb):
#         self.remove_hooks()
#
#     def _make_hook(self, module_name):
#         def hook(module, input_args, output_tensor):
#             # Parse inputs shapes
#             in_shapes = []
#             if isinstance(input_args, tuple):
#                 for arg in input_args:
#                     if torch.is_tensor(arg):
#                         in_shapes.append(list(arg.shape))
#
#             # Parse outputs shapes (handle tuple outputs common in HF models)
#             out_shapes = []
#             if isinstance(output_tensor, tuple):
#                 for out in output_tensor:
#                     if torch.is_tensor(out):
#                         out_shapes.append(list(out.shape))
#             elif torch.is_tensor(output_tensor):
#                 out_shapes.append(list(output_tensor.shape))
#
#             self.flow.append({
#                 "step": len(self.flow) + 1,
#                 "name": module_name,
#                 "class": module.__class__.__name__,
#                 "input_shapes": in_shapes,
#                 "output_shapes": out_shapes
#             })
#
#         return hook
#
#     def register(self, model):
#         """Attaches tracking hooks to all bottom-level execution layers."""
#         for name, module in model.named_modules():
#             # Only track leaf modules (actual operations) to keep the flow readable
#             if len(list(module.children())) == 0:
#                 h = module.register_forward_hook(self._make_hook(name))
#                 self.hooks.append(h)
#
#     def remove_hooks(self):
#         for h in self.hooks:
#             h.remove()
#         self.hooks = []
#
#     def print_flow(self):
#         print(f"{'STEP':<5} | {'MODULE PATH':<65} | {'INPUT SHAPES':<25} -> {'OUTPUT SHAPES'}")
#         print("-" * 125)
#         for item in self.flow:
#             print(
#                 f"{item['step']:<5} | {item['name']:<65} | {str(item['input_shapes']):<25} -> {item['output_shapes']}")



class ExecutionFlowTracker:
    def __init__(self):
        self.flow = []
        self.hooks = []
        # Maps a tensor's memory ID to the module path that produced it
        self.tensor_producers = {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def _make_hook(self, module_name):
        def hook(module, input_args, output_tensor):
            # 1. Track incoming tensor memory IDs and find who produced them
            input_sources = []
            if isinstance(input_args, tuple):
                for arg in input_args:
                    if torch.is_tensor(arg):
                        tensor_id = id(arg)
                        # Look up if a previous module in our execution loop built this tensor
                        source_module = self.tensor_producers.get(tensor_id, "External Input / Raw Embedding")
                        input_sources.append((list(arg.shape), source_module))

            # 2. Track outgoing tensor memory IDs and register this module as the producer
            output_details = []
            if isinstance(output_tensor, tuple):
                for out in output_tensor:
                    if torch.is_tensor(out):
                        tensor_id = id(out)
                        self.tensor_producers[tensor_id] = module_name
                        output_details.append(list(out.shape))
            elif torch.is_tensor(output_tensor):
                tensor_id = id(output_tensor)
                self.tensor_producers[tensor_id] = module_name
                output_details.append(list(output_tensor.shape))

            self.flow.append({
                "step": len(self.flow) + 1,
                "name": module_name,
                "class": module.__class__.__name__,
                "inputs": input_sources,
                "outputs": output_details
            })

        return hook

    def register(self, model):
        """Attaches mapping hooks to all bottom-level execution layers."""
        for name, module in model.named_modules():
            # Focus on leaf modules (actual operations) to keep the pipeline clear
            if len(list(module.children())) == 0:
                h = module.register_forward_hook(self._make_hook(name))
                self.hooks.append(h)

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()
        self.hooks = []

    def print_linked_flow(self):
        print(f"{'STEP':<5} | {'MODULE PATH':<60} | {'FEEDS FROM / CONSUMES OUTPUT OF'}")
        print("═" * 135)
        for item in self.flow:
            print(f"{item['step']:<5} | {item['name']:<60} |")

            # Print each input tensor shape along with its explicit source
            for shape, source in item['inputs']:
                print(f"{'':<8} ↳ Consumes shape {str(shape):<18} from ──> [{source}]")

            # Print what this layer outputs to the wider graph
            print(f"{'':<8} ➜ Yields shape   {str(item['outputs'])}")
            print("-" * 135)




def ExecutionFlowWrapper(processor, model, sample_row):
    # 1. Extract the raw image path directly from your dataset row format
    messages = sample_row["messages"]
    img_path = None
    user_text_prompt = ""

    # Extract the user text and image path from the structured message
    for message in messages:
        if message["role"] == "user":
            for content in message["content"]:
                if content["type"] == "image":
                    img_path = content["image"]
                elif content["type"] == "text":
                    user_text_prompt = content["text"]

    # 2. Open the image file explicitly
    if img_path is None:
        raise ValueError("Could not find a valid image path inside the sample_row!")
    image_input = Image.open(img_path).convert("RGB")

    # 3. Formulate a clean raw prompt string exactly how Qwen expects it under the hood
    # Qwen2.5-VL uses a specific tag placeholder sequence for image inputs
    raw_prompt = f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{user_text_prompt}<|im_end|>\n<|im_start|>assistant\n"

    print("🔧 Formatting inputs manually to bypass chat-template index errors...")

    # 4. Pass the raw text prompt and the explicit image list directly to the processor
    processed_inputs = processor(
        text=[raw_prompt],
        images=[image_input],
        padding=True,
        return_tensors="pt"
    )

    # 5. Move tensors to the model's active device without touching weights
    model_device = next(model.parameters()).device
    processed_inputs = {k: v.to(model_device) for k, v in processed_inputs.items()}

    # 6. Run the forward pass with our flow tracker attached
    with ExecutionFlowTracker() as tracker:
        tracker.register(model)

        print("🚀 Running tracker dry-run pass on hf_dataset[0]...")
        with torch.no_grad():
            _ = model(**processed_inputs)

        # 7. Print out the dynamic sequential execution flow map
        tracker.print_linked_flow()

