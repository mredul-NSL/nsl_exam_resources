# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 13:31:19 2021

@author: Mredul
"""
import sys
import os 

root_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(root_dir, 'musicplayer'))
import musicplayer
from musicplayer import audio, bar, foo
from musicplayer.audio import audio