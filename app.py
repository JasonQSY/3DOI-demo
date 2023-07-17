import numpy as np
import gradio as gr
import argparse
import pdb
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
from PIL import Image
import os
import subprocess
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('agg')

from monoarti.model import build_demo_model
from monoarti.detr.misc import interpolate
from monoarti.vis_utils import draw_properties, draw_affordance, draw_localization
from monoarti.detr import box_ops
from monoarti import axis_ops, depth_ops


mask_source_draw = "draw a mask on input image"
mask_source_segment = "type what to detect below"

def change_radio_display(task_type, mask_source_radio):
    text_prompt_visible = True
    inpaint_prompt_visible = False
    mask_source_radio_visible = False
    if task_type == "inpainting":
        inpaint_prompt_visible = True
    if task_type == "inpainting" or task_type == "remove":
        mask_source_radio_visible = True   
        if mask_source_radio == mask_source_draw:
            text_prompt_visible = False
    return  gr.Textbox.update(visible=text_prompt_visible), gr.Textbox.update(visible=inpaint_prompt_visible), gr.Radio.update(visible=mask_source_radio_visible)

os.makedirs('temp', exist_ok=True)

# initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
model = build_demo_model().to(device)
checkpoint_path = 'checkpoint_20230515.pth'
if not os.path.exists(checkpoint_path):
    print("get {}".format(checkpoint_path))
    result = subprocess.run(['wget', 'https://fouheylab.eecs.umich.edu/~syqian/3DOI/{}'.format(checkpoint_path)], check=True)
    print('wget {} result = {}'.format(checkpoint_path, result))    
loaded_data = torch.load(checkpoint_path, map_location=device)
state_dict = loaded_data["model"]
model.load_state_dict(state_dict, strict=True)

data_transforms = transforms.Compose([
    transforms.Resize((768, 1024)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

movable_imap = {
    0: 'one_hand',
    1: 'two_hands',
    2: 'fixture',
    -100: 'n/a',
}

rigid_imap = {
    1: 'yes',
    0: 'no',
    2: 'bad',
    -100: 'n/a',
}

kinematic_imap = {
    0: 'freeform',
    1: 'rotation',
    2: 'translation',
    -100: 'n/a'
}

action_imap = {
    0: 'free',
    1: 'pull',
    2: 'push',
    -100: 'n/a',
}




def run_model(input_image):
    image = input_image['image']
    input_width, input_height = image.size
    image_tensor = data_transforms(image)
    image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    mask = np.array(input_image['mask'])[:, :, :3].sum(axis=2)
    if mask.sum() == 0:
        raise gr.Error("No query point! Please click on the image to create a query point.")
    ret, thresh = cv2.threshold(mask.astype(np.uint8), 50, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    M = cv2.moments(contours[0])
    x = round(M['m10'] / M['m00'] / input_width * 1024) # width
    y = round(M['m01'] / M['m00'] / input_height * 768) # height
    keypoints = torch.ones((1, 15, 2)).long() * -1
    keypoints[:, :, 0] = x
    keypoints[:, :, 1] = y
    keypoints = keypoints.to(device)

    valid = torch.zeros((1, 15)).bool()
    valid[:, 0] = True
    valid = valid.to(device)

    out = model(image_tensor, valid, keypoints, bbox=None, masks=None, movable=None, rigid=None, kinematic=None, action=None, affordance=None, affordance_map=None, depth=None, axis=None, fov=None, backward=False)

    # visualization
    rgb = np.array(image.resize((1024, 768)))
    image_size = (768, 1024)
    bbox_preds = out['pred_boxes']
    mask_preds = out['pred_masks']
    mask_preds = interpolate(mask_preds, size=image_size, mode='bilinear', align_corners=False)
    mask_preds = mask_preds.sigmoid() > 0.5
    movable_preds = out['pred_movable'].argmax(dim=-1)
    rigid_preds = out['pred_rigid'].argmax(dim=-1)
    kinematic_preds = out['pred_kinematic'].argmax(dim=-1)
    action_preds = out['pred_action'].argmax(dim=-1)
    axis_preds = out['pred_axis']
    depth_preds = out['pred_depth']
    affordance_preds = out['pred_affordance']
    affordance_preds = interpolate(affordance_preds, size=image_size, mode='bilinear', align_corners=False)
    if depth_preds is not None:
        depth_preds = interpolate(depth_preds, size=image_size, mode='bilinear', align_corners=False)
    i = 0
    instances = []

    predictions = []
    for j in range(15):
        if not valid[i, j]:
            break
        export_dir = './temp'
        img_name = 'temp'

        axis_center = box_ops.box_xyxy_to_cxcywh(bbox_preds[i]).clone()
        axis_center[:, 2:] = axis_center[:, :2]
        axis_pred = axis_preds[i]
        axis_pred_norm = F.normalize(axis_pred[:, :2])
        axis_pred = torch.cat((axis_pred_norm, axis_pred[:, 2:]), dim=-1)
        src_axis_xyxys = axis_ops.line_angle_to_xyxy(axis_pred, center=axis_center)

        # original image + keypoint
        vis = rgb.copy()
        kp = keypoints[i, j].cpu().numpy()
        vis = cv2.circle(vis, kp, 24, (255, 255, 255), -1)
        vis = cv2.circle(vis, kp, 20, (31, 73, 125), -1)
        vis = Image.fromarray(vis)
        predictions.append(vis)

        # physical properties
        movable_pred = movable_preds[i, j].item()
        rigid_pred = rigid_preds[i, j].item()
        kinematic_pred = kinematic_preds[i, j].item()
        action_pred = action_preds[i, j].item()
        output_path = os.path.join(export_dir, '{}_kp_{:0>2}_02_phy.png'.format(img_name, j))
        draw_properties(output_path, movable_pred, rigid_pred, kinematic_pred, action_pred)
        property_pred = Image.open(output_path)
        predictions.append(property_pred)

        # box mask axis
        axis_pred = src_axis_xyxys[j]
        if kinematic_imap[kinematic_pred] != 'rotation':
            axis_pred = [-1, -1, -1, -1]
        img_path = os.path.join(export_dir, '{}_kp_{:0>2}_03_loc.png'.format(img_name, j))
        draw_localization(
            rgb, 
            img_path, 
            None,
            mask_preds[i, j].cpu().numpy(),
            axis_pred,
            colors=None,
            alpha=0.6,    
        )
        localization_pred = Image.open(img_path)
        predictions.append(localization_pred)

        # affordance
        affordance_pred = affordance_preds[i, j].sigmoid()
        affordance_pred = affordance_pred.detach().cpu().numpy() #[:, :, np.newaxis]
        aff_path = os.path.join(export_dir, '{}_kp_{:0>2}_04_affordance.png'.format(img_name, j))
        aff_vis = draw_affordance(rgb, aff_path, affordance_pred)
        predictions.append(aff_vis)

        # depth
        depth_pred = depth_preds[i]
        depth_pred_metric = depth_pred[0] * 0.945 + 0.658
        depth_pred_metric = depth_pred_metric.detach().cpu().numpy()
        fig = plt.figure()
        plt.imshow(depth_pred_metric, cmap=mpl.colormaps['plasma'])
        plt.axis('off')
        depth_path = os.path.join(export_dir, '{}_kp_{:0>2}_05_depth.png'.format(img_name, j))
        plt.savefig(depth_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        depth_pred = Image.open(depth_path)
        predictions.append(depth_pred)

    return predictions


examples = [
    'examples/AR_4ftr44oANPU_34_900_35.jpg',
    'examples/AR_0Mi_dDnmF2Y_6_2610_15.jpg',
    'examples/EK_0037_P28_101_frame_0000031096.jpg',
    'examples/EK_0056_P04_121_frame_0000018401.jpg',
    'examples/taskonomy_bonfield_point_42_view_6_domain_rgb.png',
    'examples/taskonomy_wando_point_156_view_3_domain_rgb.png',
]

title = 'Understanding 3D Object Interaction from a Single Image'
authors = """
<p style='text-align: center'> <a href='https://jasonqsy.github.io/3DOI/' target='_blank'>Project Page</a> | <a href='https://arxiv.org/abs/2305.09664' target='_blank'>Paper</a> | <a href='https://github.com/JasonQSY/3DOI' target='_blank'>Code</a></p>
"""
description = """
Gradio demo for Understanding 3D Object Interaction from a Single Image. \n
You may click on of the examples or upload your own image. \n
After having the image, you can click on the image to create a single query point. You can then hit Run.\n
Our approach can predict 3D object interaction from a single image, including Movable (one hand or two hands), Rigid, Articulation type and axis, Action, Bounding box, Mask, Affordance and Depth.\n
Since the demo is run on cpu, it needs approximately 30 seconds to inference, which is slow. You can either fork the huggingface space, or visit https://openxlab.org.cn/apps/detail/JasonQSY/3DOI for the same demo with Nvidia A10G.
"""

def change_language(lang_select, description_controller, run_button):
    description_cn = """
要运行demo，首先点击右边的示例图片或者上传自己的图片。在有了图片以后，点击图片上的点来创建query point，然后点击 Run。
"""
    if lang_select == "简体中文":
        description_controller = description_cn
        run_button = '运行'
    else:
        description_controller = description
        run_button = 'Run'
    
    return description_controller, run_button
    

with gr.Blocks().queue() as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>")
    gr.Markdown(authors)
    # gr.Markdown("<p style='text-align: center'>ICCV 2023</p>")
    

    lang_select = gr.Dropdown(["简体中文", "English"], label='Language / 语言')
    
    description_controller = gr.Markdown(description)
    

    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(source='upload', elem_id="image_upload", tool='sketch', type='pil', label="Upload", brush_radius=20)  
            run_button = gr.Button(label="Run")

        with gr.Column():
            examples_handler = gr.Examples(
                examples=examples,
                inputs=input_image,
                examples_per_page=10,
            )

    with gr.Row():
        with gr.Column(scale=1):
                query_image = gr.outputs.Image(label="Image + Query", type="pil")

        with gr.Column(scale=1):
            pred_localization = gr.outputs.Image(label="Localization", type="pil")

        with gr.Column(scale=1):
            pred_properties = gr.outputs.Image(label="Properties", type="pil")

    with gr.Row():
        with gr.Column(scale=1):
            pred_affordance = gr.outputs.Image(label="Affordance", type="pil")

        with gr.Column(scale=1):
            pred_depth = gr.outputs.Image(label="Depth", type="pil")

        with gr.Column(scale=1):
            pass

    lang_select.change(
        change_language,
        inputs=[lang_select, description_controller, run_button], 
        outputs=[description_controller, run_button]
    )

    output_components = [query_image, pred_properties, pred_localization, pred_affordance, pred_depth]
    run_button.click(fn=run_model, inputs=[input_image], outputs=output_components)


if __name__ == "__main__":
    demo.launch(server_name='0.0.0.0')