# sd-webui-video-sa
Shifting attention in the prompt over the duration of a video

# Usage

Upload video.

Make sure to put an image with same size in the regular image-box. Without this you'll get an error.

Generate.

Result should show up in img2img folder under video_shift.

# Example

Prompt: (8 bit retro style:0\~0\~1.3) (cyberpunk neon vaporwave:1.3\~0\~0), (cool cyberpunk:1\~0) (8 bit game, nintendo:0\~1) man, (sparkles:0\~1.4\~0)
Negative prompt: cyber, sketch, text, watermark, (glasses:1.3), mask, headphones, woman, girl, (cyberpunk neon vaporwave:0\~1)
Steps: 20 | Sampler: DPM++ 2M SDE Karras | CFG scale: 5.5 | Seed: 1989035025 | Size: 640x360 | Model hash: bfea7e18e2 | Model: absolutereality_v10 | VAE: rmada-cold-vae | Denoising strength: 0.36 | Clip skip: 1 | Version: unknown | Token merging ratio: 0.5 | Parser: Full parser | Eta: 0

https://github.com/yownas/sd-webui-video-sa/assets/13150150/5158024c-4df2-427c-94d6-5b7b1e1e8171

# TODO

* Copy audio
* See if there is a way to not require junk image 

