import gradio as gr
import os
from PIL import Image
import json
import re
import random
def remove_extra_commas(json_str):
    fixed_json = re.sub(r',\s*([\]}])', r'\1', json_str)
    return fixed_json

def convert_label(cls_select):
    if cls_select == "vru":
        return "vulnerable_road_users"
    elif cls_select == "traffic_signs":
        return "traffic signs"
    elif cls_select == "traffic_lights":
        return "traffic lights"
    elif cls_select == "traffic_cones":
        return "traffic cones"
    elif cls_select == "other_objects":
        return "other objects"
    else:
        return cls_select
    
def show(text, cls_select):
    image_name = text.zfill(4) + '.jpg'
    new_label = convert_label(cls_select)
    new_imgage_name = image_name.split('.')[0] + '_' + cls_select + '.jpg'
    current_file_path = os.path.abspath(__file__)
    current_directory = os.path.dirname(current_file_path)
    img1 = Image.open(os.path.join(current_directory, "images", image_name))
    img2 = Image.open(os.path.join(current_directory, "vis_images", new_imgage_name))
    txt_name_1 = os.path.join(current_directory, "annotations", "first", image_name.split('.')[0]+'.json')
    txt_name_2 = os.path.join(current_directory, "annotations", "second", image_name.split('.')[0]+'.json')
    txt_name_3 = os.path.join(current_directory, "annotations", "third", image_name.split('.')[0]+'.json')

    
    custum_sum = 0
    info_list1 = []
    info_list2 = []
    info_list3 = []
    info_list = []
    if os.path.exists(txt_name_1):
        with open(txt_name_1, 'r') as f:
            parsed_json = json.load(f)
            if new_label in parsed_json.keys():
                info = parsed_json[new_label]
            else:
                info = parsed_json[cls_select]
            if new_label == 'suggestions':
                info_list1.append("V1" + "\n" + info )
            else:
                filtered_info = [d for d in info if d]
                for idx, item in enumerate(filtered_info):
                    if idx == 0:
                        info_list1.append("V1" + "\n" + "description: " + item['description'] + "\n" + "explanation: " + item['explanation'])
                    else:
                        info_list1.append("description: " + item['description'] + "\n" + "explanation: " + item['explanation'])
                custum_sum += len(filtered_info)

    if os.path.exists(txt_name_2):
        with open(txt_name_2, 'r') as f:
            parsed_json = json.load(f)
            # info = parsed_json[new_label]
            if new_label in parsed_json.keys():
                info = parsed_json[new_label]
            else:
                info = parsed_json[cls_select]
        if new_label == 'suggestions':
                info_list2.append("V2" + "\n" + info)
        else:
            # List[Dict] assert Dict has key 'description' and 'explanation'
            filtered_info = [d for d in info if d]
            for idx, item in enumerate(filtered_info):
                if idx == 0:
                    info_list2.append("V2" + "\n" + "description: " + item['description'] + "\n" + "explanation: " + item['explanation'])
                else:
                    info_list2.append("description: " + item['description'] + "\n" + "explanation: " + item['explanation'])
            custum_sum += len(filtered_info)

    if os.path.exists(txt_name_3):
        with open(txt_name_3, 'r') as f:
            parsed_json = json.load(f)
            # info = parsed_json[new_label]
            if new_label in parsed_json.keys():
                info = parsed_json[new_label]
            else:
                info = parsed_json[cls_select]
        if new_label == 'suggestions':
                info_list3.append("V3" + "\n" + info)
        else:
            # List[Dict] assert Dict has key 'description' and 'explanation'
            filtered_info = [d for d in info if d]
            for idx, item in enumerate(filtered_info):
                if idx == 0:
                    info_list3.append("V3" + "\n" + "description: " + item['description'] + "\n" + "explanation: " + item['explanation'])
                else:
                    info_list3.append("description: " + item['description'] + "\n" + "explanation: " + item['explanation'])
            custum_sum += len(filtered_info)
        info_list.append(info_list1)
        info_list.append(info_list2)
        info_list.append(info_list3)
        indexed_info_list = list(enumerate(info_list))
        random.shuffle(indexed_info_list)
        index_list_shuffle = [item for item, _ in indexed_info_list]
        save_path = os.path.join(os.path.dirname(current_file_path), "refine_result_ttt", text.zfill(4))
        os.makedirs(save_path, exist_ok=True)
        index_text_name = 'index_' + cls_select + '.txt'
        with open(os.path.join(save_path, index_text_name), 'w') as fp:
            for text_index in index_list_shuffle:
                fp.write(str(text_index + 1) + ' ')
        info_list_shuffle = [item for _, sublist in indexed_info_list for item in sublist]
    
    remove_list = [""] * (14 - len(info_list_shuffle))
    return img1, img2, *info_list_shuffle, *remove_list

def save_text(d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14, final_dro, text, cls_select):
    d_list = [d1, d2, d3, d4, d5, d6, d7, d8, d9, d10, d11, d12, d13, d14]
    good_text = ""
    modify_text = ""
    for d in d_list:
        if d == "good":
            good_text = good_text + '1 ' 
        else:
            good_text = good_text + '0 '
        if d == 'modify':
            modify_text = modify_text +'1 '
        else:
            modify_text = modify_text +'0 '
    if final_dro == "complete":
        complete_info = 1
    else:
        complete_info = 0
    complete_text_name = 'complete_' + cls_select +'.txt'
    good_txt_name = 'good_' + cls_select + '.txt'
    modify_txt_name = 'modify_' + cls_select + '.txt'
    current_file_path = os.path.abspath(__file__)
    save_path = os.path.join(os.path.dirname(current_file_path), "refine_result", text.zfill(4))
    os.makedirs(save_path, exist_ok=True)
    with open(os.path.join(save_path, good_txt_name), 'w') as fp:
        fp.write(good_text)
    with open(os.path.join(save_path, modify_txt_name), 'w') as fp:
        fp.write(modify_text)
    with open(os.path.join(save_path, complete_text_name), 'w') as fp:
        fp.write(str(complete_info))
    return "Submit successfully!"

rules_markdown = """ ### Guidelines
- Input the image number (ranging from 1 to 4884) along with the specific category you wish to examine, then press "Display" to reveal both the image and its associated annotation data.
- The original image will appear on the left side, whereas the visualization pertaining to the chosen category will be shown on the right.
- There are no visualizations for traffic sign and traffic light; the visualization results for barriers and miscellaneous are identical.
- According to the annotation rules, select "good", "modify", or "delete" for each text box.
- Based on the selected good and modify annotations, assess whether the image completely describes the salient objects of that category, marking it as either "complete" or "incomplete".
- After making your selections, click "Submit". A "Submit successfully!" message will display upon successful submission.
- Click "Clear" to refresh the page and start a new round.
"""

with gr.Blocks() as demo:
    gr.Markdown(rules_markdown)
    with gr.Row():\
        
        with gr.Column(scale=20):
            text = gr.Textbox(
                label="Image name",
                placeholder="Input image name: 1"
            )
        with gr.Column(scale=5):
            cls_select = gr.Dropdown(["vehicles", "vru", "traffic_signs", "traffic_lights", "traffic_cones", "barriers", "other_objects", "suggestions"], label="Class selection")
    with gr.Row():
        display_btn = gr.Button(value="Display")

    with gr.Row():
        with gr.Column(scale=1):
            img1 = gr.Image()
        with gr.Column(scale=1):
            img2 = gr.Image()

    with gr.Column():
        with gr.Row():
            text1 = gr.Textbox(scale=3, show_label = False)
            tie_dro1 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text2 = gr.Textbox(scale=3, show_label = False)
            tie_dro2 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text3 = gr.Textbox(scale=3, show_label = False)
            tie_dro3 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text4 = gr.Textbox(scale=3, show_label = False)
            tie_dro4 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text5 = gr.Textbox(scale=3, show_label = False)
            tie_dro5 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text6 = gr.Textbox(scale=3, show_label = False)
            tie_dro6 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text7 = gr.Textbox(scale=3, show_label = False)
            tie_dro7 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text8 = gr.Textbox(scale=3, show_label = False)
            tie_dro8 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text9 = gr.Textbox(scale=3, show_label = False)
            tie_dro9 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text10 = gr.Textbox(scale=3, show_label = False)
            tie_dro10 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text11 = gr.Textbox(scale=3, show_label = False)
            tie_dro11 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text12 = gr.Textbox(scale=3, show_label = False)
            tie_dro12 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text13 = gr.Textbox(scale=3, show_label = False)
            tie_dro13 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            text14 = gr.Textbox(scale=3, show_label = False)
            tie_dro14 = gr.Dropdown(["good", "modify", "delete"], label="Label selection", scale=1)
        with gr.Row():
            final_dro = gr.Dropdown(["complete","incomplete"], label="Ann Complete choice", scale=2)
            btn = gr.Button("Submit", scale=2)
            text_output = gr.Textbox(label="Output", scale=2)
        clear_btn = gr.Button(value="🗑️ Clear")

    img_list = [img1, img2]
    text_list = [text1, text2, text3, text4, text5, text6, text7, text8, text9, text10, text11, text12, text13, text14]
    btn_list = [display_btn, btn, clear_btn]
    drop_down_list = [tie_dro1, tie_dro2, tie_dro3, tie_dro4, tie_dro5, tie_dro6, tie_dro7, tie_dro8, tie_dro9, tie_dro10, tie_dro11, tie_dro12, tie_dro13, tie_dro14, final_dro]
    display_btn.click(show, inputs=[text, cls_select], outputs=img_list + text_list)   
    btn.click(save_text, inputs=drop_down_list + [text, cls_select], outputs=text_output)
    
    clear_btn.click(lambda: ["", None, "", None, None, "", "", "", "", "", "", "", "", "", "", "", "", "", "",
        None, None, None, None, None, None, None, None, None, None, None, None, None, None, None, "Display", "Submit", "Clear"], 
        None, [text, cls_select, text_output] + img_list + text_list + drop_down_list + btn_list)

if __name__ == "__main__":
    demo.launch(share=True)