# Video Shift Attention script
#
# https://github.com/yownas/
#
# Give a prompt like: "photo of (cat:1~0) or (dog:0~1)"
# Generates a sequence of images, lowering the weight of "cat" from 1 to 0 and increasing the weight of "dog" from 0 to 1.
# Will also support multiple numbers. "(cat:1~0~1)" will go from cat:1 to cat:0 to cat:1 streched over the input video.

import gradio as gr
import imageio
import math
import numpy as np
import os
from PIL import Image
import random
import re
import types
import modules.scripts as scripts
from modules.processing import Processed, process_images, fix_seed
from modules.shared import opts, cmd_opts, state

import cv2

class Script(scripts.Script):
    def title(self):
        return "Video shift attention"

    def show(self, is_img2img):
        return is_img2img

    def ui(self, is_img2img):
        #FIXME add text about junk-image
        input_video = gr.File(label="Upload video", visible=True, file_types=['.mp4'], file_count = "single")
        #FIXME: add show_images

        return [input_video]

    def get_next_sequence_number(path):
        from pathlib import Path
        """
        Determines and returns the next sequence number to use when saving an image in the specified directory.
        The sequence starts at 0.
        """
        result = -1
        dir = Path(path)
        for file in dir.iterdir():
            if not file.is_dir(): continue
            try:
                num = int(file.name)
                if num > result: result = num
            except ValueError:
                pass
        return result + 1

    def run(self, p, input_video):


        #FIXME
        show_images = False


        re_attention_span = re.compile(r"([\-.\d]+~[\-~.\d]+)", re.X)

        def shift_attention(text, distance):

            def inject_value(distance, match_obj):
                a = match_obj.group(1).split('~')
                l = len(a) - 1
                q1 = int(math.floor(distance*l))
                q2 = int(math.ceil(distance*l))
                return str( float(a[q1]) + ((float(a[q2]) - float(a[q1])) * (distance * l - q1)) )

            res = re.sub(re_attention_span, lambda match_obj: inject_value(distance, match_obj), text)
            return res

        initial_info = None
        images = []
        dists = []

        if not input_video:
            print(f"Nothing to do. Please upload a video.")
            return Processed(p, images, p.seed)

        # Custom folder for saving images/animations
        shift_path = os.path.join(p.outpath_samples, "video_shift")
        os.makedirs(shift_path, exist_ok=True)
        shift_number = Script.get_next_sequence_number(shift_path)
        shift_path = os.path.join(shift_path, f"{shift_number:05}")
        os.makedirs(shift_path, exist_ok=True)
        p.outpath_samples = shift_path

        # Force Batch Count and Batch Size to 1.
        p.n_iter = 1
        p.batch_size = 1

        # Make sure seed is fixed
        fix_seed(p)

        initial_prompt = p.prompt
        initial_negative_prompt = p.negative_prompt
        initial_seed = p.seed
        cfg_scale = p.cfg_scale

        # Kludge for seed travel 
        p.subseed = p.seed

        # Split prompt and generate list of prompts
        promptlist = re.split("(THEN\([^\)]*\)|THEN)", p.prompt)+[None]
        negative_promptlist = re.split("(THEN\([^\)]*\)|THEN)", p.negative_prompt)+[None]

        # Build new list
        prompts = []
        while len(promptlist) or len(negative_promptlist):
            prompt, subseed, negprompt, negsubseed, new_cfg_scale = (None, None, None, None, None)

            if len(negative_promptlist):
                negprompt = negative_promptlist.pop(0).strip()
                opts = negative_promptlist.pop(0)

                if opts:
                    opts = re.sub('THEN\((.*)\)', '\\1', opts)
                    opts = None if opts == 'THEN' else opts
                    if opts:
                        for then_data in opts.split(','): # Get values from THEN()
                            if '=' in then_data:
                                opt, val = then_data.split('=')
                                if opt == 'seed':
                                    try:
                                        negsubseed = int(val)
                                    except:
                                        negsubseed = None
                                if opt == 'cfg':
                                    try:
                                        new_cfg_scale = float(val)
                                    except:
                                        new_cfg_scale = None

            if len(promptlist):
                prompt = promptlist.pop(0).strip() # Prompt
                opts = promptlist.pop(0) # THEN()
                if opts:
                    opts = re.sub('THEN\((.*)\)', '\\1', opts)
                    opts = None if opts == 'THEN' else opts
                    if opts:
                        for then_data in opts.split(','): # Get values from THEN()
                            if '=' in then_data:
                                opt, val = then_data.split('=')
                                if opt == 'seed':
                                    try:
                                        subseed = int(val)
                                    except:
                                        subseed = None
                                if opt == 'cfg':
                                    try:
                                        new_cfg_scale = float(val)
                                    except:
                                        new_cfg_scale = None

            if not subseed:
                subseed = negsubseed
            prompts += [(prompt, negprompt, subseed, new_cfg_scale)]

        input_images = [] 
        video_read = cv2.VideoCapture(input_video.name)
        video_fps = video_read.get(cv2.CAP_PROP_FPS)

        while(True):
            ret,frame = video_read.read()
            if ret:
                input_images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            else:
                break

        # Set generation helpers
        total_images = len(input_images)
        steps = int(total_images / len(prompts))
        state.job_count = total_images
        print(f"Generating {total_images} images.")

        # Generate prompt_images and add to images (the big list)
        prompt = p.prompt
        negprompt = p.negative_prompt
        seed = p.seed
        subseed = p.subseed
        cfg_scale = p.cfg_scale
        step = 0
        for new_prompt, new_negprompt, new_subseed, new_cfg_scale in prompts:
            if new_prompt: 
                prompt = new_prompt
            if new_negprompt:
                negprompt = new_negprompt
            if new_subseed:
                subseed = new_subseed

            p.seed = seed
            p.subseed = subseed

            # Frames for the current prompt pair
            prompt_images = []
            dists = []

            # Empty prompt
            if not new_prompt and not new_negprompt: 
                #print("NO PROMPT")
                break

            #DEBUG
            print(f"Shifting prompt:\n+ {prompt}\n- {negprompt}\nSeeds: {int(seed)}/{int(subseed)} CFG: {cfg_scale}~{new_cfg_scale}")

            # Generate the steps
            for i in range(int(steps) + 1):
                if state.interrupted:
                    break

                p.init_images = [input_images[min(step, len(input_images)-1)]]
                step += 1
    
                distance = float(i / int(steps))
                p.prompt = shift_attention(prompt, distance)
                p.negative_prompt = shift_attention(negprompt, distance)
                p.subseed_strength = distance
                if isinstance(new_cfg_scale, types.FloatType):
                    p.cfg_scale = cfg_scale * (1.-distance) + new_cfg_scale * distance

                proc = process_images(p)

                if initial_info is None:
                    initial_info = proc.info
    
                image = [proc.images[0]]
    
                prompt_images += image
                dists += [distance]

            # We should have reached the subseed if we were seed traveling
            seed = subseed
            # ..and the cfg
            if new_cfg_scale:
                cfg_scale = new_cfg_scale

            # End of prompt_image loop
            images += prompt_images

        # Save video
        try:
            frames = [np.asarray(t) for t in images]
            fps = video_fps if video_fps > 0 else len(frames) / abs(video_fps)
            filename = f"videoshift-{shift_number:05}.mp4"
            writer = imageio.get_writer(os.path.join(shift_path, filename), fps=fps)
            for frame in frames:
                writer.append_data(frame)
            writer.close()
        except Exception as err:
            print(f"ERROR: Failed generating video: {err}")

        processed = Processed(p, images if show_images else [], p.seed, initial_info)

        return processed

    def describe(self):
        return "Shift attention for videos."
