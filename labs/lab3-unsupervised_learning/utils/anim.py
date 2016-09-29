# based on http://jakevdp.github.io/blog/2013/05/12/embedding-matplotlib-animations/
import matplotlib.pyplot as plt

#for windows support:
import os
import sys
import string
import random

import matplotlib.animation as animation
import base64

from tempfile import NamedTemporaryFile
from IPython.display import HTML

VIDEO_TAG = """<video controls>
 <source src="data:video/x-m4v;base64,{0}" type="video/mp4">
 Your browser does not support the video tag.
</video>"""

if sys.platform in ['cygwin','win32']:
    def anim_to_html(anim,fps=1):
        if not hasattr(anim, '_encoded_video'):
            N = 10 # length of fileID
            fileID = ''.join([random.choice(string.ascii_letters) for i in range(N)])
            anim.save('./utils/'+fileID+".mp4", fps=fps, extra_args=['-vcodec', 'libx264'])

            video = open('./utils/'+fileID+".mp4","rb").read()

        anim._encoded_video = base64.b64encode(video).decode('utf-8')
        return VIDEO_TAG.format(anim._encoded_video)
else:
    def anim_to_html(anim,fps=1):
        if not hasattr(anim, '_encoded_video'):
            with NamedTemporaryFile(suffix='.mp4') as f:
                anim.save(f.name, fps=fps,
                      extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
                video = open(f.name, "rb").read()
            anim._encoded_video = base64.b64encode(video).decode('utf-8')

        return VIDEO_TAG.format(anim._encoded_video)

def display_animation(anim,fps=1):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim,fps=fps))

def init():
    return

def clearFiles():
    [os.remove('./utils/'+f) for f in os.listdir('./utils') if f.endswith('.mp4')]
    return
