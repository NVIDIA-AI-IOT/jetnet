import os
from jetnet.utils import import_object, make_parent_dir
from jetnet.all_configs import (
    ALL_CONFIGS,
    IMAGE_DATASET_CONFIGS,
    CLASSIFICATION_CONFIGS,
    DETECTION_CONFIGS,
    TEXT_DETECTION_CONFIGS,
    POSE_CONFIGS
)


def define_env(env):

    @env.macro
    def render_configs(config_strs):
        s = "| Class | Name | Config |\n"
        s += "|------|------|--------|\n"
        for config_str in config_strs:
            try:
                config = import_object(config_str)
                config_path = os.path.join('/configs/', '/'.join(config_str.split('.')) + '.json')
                s += f"| {config.__class__.__name__} | {config_str} | [json]({config_path}) |\n"
            except:
                print(f"could not import {config_str}")
        s += "\n"
        return s

    @env.macro
    def render_image_dataset_configs():
        return render_configs(IMAGE_DATASET_CONFIGS)

    @env.macro
    def render_classification_configs():
        return render_configs(CLASSIFICATION_CONFIGS)

    @env.macro
    def render_detection_configs():
        return render_configs(DETECTION_CONFIGS)

    @env.macro
    def render_text_detection_configs():
        return render_configs(TEXT_DETECTION_CONFIGS)

    @env.macro
    def render_pose_configs():
        return render_configs(POSE_CONFIGS)



def write_configs(root):
    "Post-build actions"
    
    for config_str in ALL_CONFIGS:
        try:
            config = import_object(config_str)
            path = os.path.join(root, '/'.join(config_str.split('.'))) + ".json"
            make_parent_dir(path)
            
            with open(path, 'w') as f:
                f.write(config.json(indent=2))
        except:
            print(f"could not import {config_str}")

def on_post_build(env):
    "Post-build actions"
    site_dir = env.conf['site_dir']
    write_configs(os.path.join(site_dir, "configs"))