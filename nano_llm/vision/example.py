#!/usr/bin/env python3
#
# This multimodal example is a simplified version of the 'Live Llava' demo,
# wherein the same prompt (or set of prompts) is applied to a stream of images.
#
# You can run it like this (these options will replicate the defaults)
#
#    python3 -m nano_llm.vision.example \
#      --model Efficient-Large-Model/VILA-2.7b \
#      --video-input "/data/images/*.jpg" \
#      --prompt "Describe the image." \
#      --prompt "Are there people in the image?"
#
# You can specify multiple prompts (or a text file) to be applied to each image,
# and the video inputs can be sequences of files, camera devices, or network streams.
# For example, `--video-input /dev/video0` will capture from a V4L2 webcam. See here:
#  https://github.com/dusty-nv/jetson-inference/blob/master/docs/aux-streaming.md
#
import os
from nano_llm import NanoLLM, ChatHistory
from nano_llm.utils import ArgParser, load_prompts
from nano_llm.plugins import VideoSource, VideoOutput
from nano_llm.web import WebServer


from termcolor import cprint

# parse args and set some defaults
args = ArgParser(extras=ArgParser.Defaults + ['prompt', 'video_input', 'video_output']).parse_args()
prompts = load_prompts(args.prompt)

if not prompts:
    prompts = ["Describe the image.", "Are there people in the image?"]
    
if not args.model:
    args.model = "Efficient-Large-Model/VILA-2.7b"

if not args.video_input:
    args.video_input = "/data/images/*.jpg"


video_output = VideoOutput("webrtc://@:8554/output")
    
print(args)

# load vision/language model
model = NanoLLM.from_pretrained(
    args.model, 
    api=args.api,
    quantization=args.quantization, 
    max_context_len=args.max_context_len,
    vision_model=args.vision_model,
    vision_scaling=args.vision_scaling, 
)

assert(model.has_vision)

# create the chat history
chat_history = ChatHistory(model, args.chat_template, args.system_prompt)

# open the video stream
video_source = VideoSource(**vars(args))

webrtc_args = {}
video_output_args = video_output.stream.GetOptions()['resource']
webrtc_args.update(dict(webrtc_output_stream=video_output_args['path'].strip('/'), 
                                webrtc_output_port=video_output_args['port']))

webrtc_args.update(dict(webrtc_input_stream=video_output_args['path'].strip('/'), 
                                webrtc_input_port=video_output_args['port']))
server = WebServer(
        # msg_callback=self.on_websocket, 
        index='video_query.html', 
        title="VILA Test", 
        # model=os.path.basename(model),
        # mounts=mounts,
        # nanodb=nanodb,
        **webrtc_args,
        # **kwargs
    )
server.start()

# apply the prompts to each frame
while True:
    img = video_source.capture()
    
    if img is None:
        continue

    out = img
    video_output.process(out)

    chat_history.append(role='user', image=img)
    
    for prompt in prompts:
        chat_history.append(role='user', msg=prompt)
        embedding, _ = chat_history.embed_chat()
        
        print('>>', prompt)
        
        reply = model.generate(
            embedding,
            kv_cache=chat_history.kv_cache,
            max_new_tokens=args.max_new_tokens,
            min_new_tokens=args.min_new_tokens,
            do_sample=args.do_sample,
            repetition_penalty=args.repetition_penalty,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        
        for token in reply:
            cprint(token, 'blue', end='\n\n' if reply.eos else '', flush=True)

        chat_history.append(role='bot', text=reply.text)
        chat_history.kv_cache = reply.kv_cache
        
    chat_history.reset()
    
    if video_source.eos:
        break